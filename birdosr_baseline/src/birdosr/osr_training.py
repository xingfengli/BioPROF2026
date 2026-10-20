from __future__ import annotations

from dataclasses import dataclass


GEOMETRY_METHODS = {"protosphere", "rpl"}
EMBEDDING_TRAINING_METHODS = {"protosphere", "rpl", "supcon"}


def _last_linear(module):
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return module
    for child in reversed(list(module.children())):
        found = _last_linear(child)
        if found is not None:
            return found
    return None


def infer_embedding_dim(model) -> int:
    head = getattr(model, "head", None)
    linear = _last_linear(head) if head is not None else None
    if linear is None:
        for name in ("fc", "classifier", "heads", "last_linear"):
            module = getattr(model, name, None)
            linear = _last_linear(module) if module is not None else None
            if linear is not None:
                break
    if linear is None:
        raise ValueError("Could not infer embedding dimension from classifier head.")
    return int(linear.in_features)


class ProtoSphereHead:
    pass


def build_protosphere_head(num_classes: int, embedding_dim: int, radius_init: float):
    import torch
    import torch.nn as nn

    class _ProtoSphereHead(nn.Module):
        osr_head_type = "protosphere"

        def __init__(self):
            super().__init__()
            self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_dim))
            self.log_radii = nn.Parameter(torch.full((num_classes,), float(radius_init)).log())
            nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

        def distances(self, embedding):
            return torch.cdist(embedding.float(), self.prototypes.float(), p=2)

        def radii(self):
            return self.log_radii.exp().clamp_min(1e-6)

    return _ProtoSphereHead()


def build_rpl_head(num_classes: int, embedding_dim: int):
    import torch
    import torch.nn as nn

    class _RPLHead(nn.Module):
        osr_head_type = "rpl"

        def __init__(self):
            super().__init__()
            self.reciprocal_points = nn.Parameter(torch.empty(num_classes, embedding_dim))
            nn.init.normal_(self.reciprocal_points, mean=0.0, std=0.02)

        def distances(self, embedding):
            return torch.cdist(embedding.float(), self.reciprocal_points.float(), p=2)

    return _RPLHead()


def attach_osr_modules(model, cfg: dict, num_classes: int):
    import torch.nn as nn

    train_cfg = cfg.get("osr_training", {})
    method = str(train_cfg.get("method", "none")).strip().lower().replace("-", "_")
    if method in {"none", ""}:
        return model

    embedding_dim = infer_embedding_dim(model)
    if method == "protosphere":
        model.osr_head = build_protosphere_head(
            num_classes,
            embedding_dim,
            radius_init=float(train_cfg.get("radius_init", 1.0)),
        )
    elif method == "rpl":
        model.osr_head = build_rpl_head(num_classes, embedding_dim)
    elif method == "supcon":
        # SupCon does not require an extra inference head, but registering an
        # identity module keeps checkpoints/configs explicit and harmless.
        model.osr_head = nn.Identity()
        model.osr_head.osr_head_type = "supcon"
    elif method == "oe":
        model.osr_head = nn.Identity()
        model.osr_head.osr_head_type = "oe"
    else:
        valid = ", ".join(["none", "protosphere", "rpl", "supcon", "oe"])
        raise ValueError(f"Unsupported osr_training.method '{method}'. Valid methods: {valid}")
    return model


def osr_training_method(cfg: dict) -> str:
    return str(cfg.get("osr_training", {}).get("method", "none")).strip().lower().replace("-", "_")


def needs_training_embedding(cfg: dict) -> bool:
    method = osr_training_method(cfg)
    return method in EMBEDDING_TRAINING_METHODS or bool(cfg.get("training", {}).get("oe", {}).get("enabled", False))


def forward_with_optional_embedding(model, inputs, need_embedding: bool):
    if need_embedding:
        if not hasattr(model, "extract_embedding") or not hasattr(model, "classify_embedding"):
            raise ValueError("This OSR training method requires a model with extract_embedding/classify_embedding.")
        embedding = model.extract_embedding(inputs)
        logits = model.classify_embedding(embedding)
        return logits, embedding
    return model(inputs), None


def protosphere_loss(model, embedding, target, cfg: dict):
    import torch
    import torch.nn.functional as F

    train_cfg = cfg.get("osr_training", {})
    head = getattr(model, "osr_head", None)
    if head is None or getattr(head, "osr_head_type", None) != "protosphere":
        return embedding.new_tensor(0.0)
    distances = head.distances(embedding)
    batch_indices = torch.arange(target.shape[0], device=target.device)
    pos = distances[batch_indices, target]
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask[batch_indices, target] = True
    neg = distances.masked_fill(mask, float("inf")).min(dim=1).values
    margin = float(train_cfg.get("prototype_margin", 1.0))
    metric_loss = F.relu(pos - neg + margin).mean()
    radius = head.radii()[target]
    radius_loss = F.relu(pos - radius).pow(2).mean()
    radius_reg = head.radii().mean()
    return (
        float(train_cfg.get("prototype_loss_weight", 0.1)) * metric_loss
        + float(train_cfg.get("radius_loss_weight", 0.05)) * radius_loss
        + float(train_cfg.get("radius_reg_weight", 1e-4)) * radius_reg
    )


def rpl_loss(model, embedding, target, cfg: dict):
    import torch
    import torch.nn.functional as F

    train_cfg = cfg.get("osr_training", {})
    head = getattr(model, "osr_head", None)
    if head is None or getattr(head, "osr_head_type", None) != "rpl":
        return embedding.new_tensor(0.0)
    distances = head.distances(embedding)
    batch_indices = torch.arange(target.shape[0], device=target.device)
    own = distances[batch_indices, target]
    reciprocal_margin = float(train_cfg.get("reciprocal_margin", 5.0))
    repel_loss = F.relu(reciprocal_margin - own).pow(2).mean()
    return float(train_cfg.get("rpl_loss_weight", 0.05)) * repel_loss


def supervised_contrastive_loss(embedding, labels, temperature: float = 0.1):
    import torch
    import torch.nn.functional as F

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embedding.device)
    features = F.normalize(embedding.float(), dim=1)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives = mask.sum(dim=1)
    valid = positives > 0
    if not valid.any():
        return embedding.new_tensor(0.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / positives[valid]
    return -mean_log_prob_pos.mean()


def auxiliary_osr_loss(model, embedding, target, cfg: dict):
    method = osr_training_method(cfg)
    if embedding is None:
        return None
    if method == "protosphere":
        return protosphere_loss(model, embedding, target, cfg)
    if method == "rpl":
        return rpl_loss(model, embedding, target, cfg)
    if method == "supcon":
        train_cfg = cfg.get("osr_training", {})
        return float(train_cfg.get("supcon_loss_weight", 0.05)) * supervised_contrastive_loss(
            embedding,
            target,
            temperature=float(train_cfg.get("supcon_temperature", 0.1)),
        )
    return embedding.new_tensor(0.0)


def oe_loss(model, logits, embedding, cfg: dict):
    import torch.nn.functional as F

    oe_cfg = cfg.get("training", {}).get("oe", {})
    if logits is None:
        raise ValueError("OE loss requires logits.")
    energy = logits.logsumexp(dim=1).mean()
    log_probs = logits.log_softmax(dim=1)
    uniform_loss = -log_probs.mean(dim=1).mean()
    loss = (
        float(oe_cfg.get("energy_weight", 0.1)) * energy
        + float(oe_cfg.get("uniform_weight", 0.1)) * uniform_loss
    )
    head = getattr(model, "osr_head", None)
    if embedding is not None and head is not None and hasattr(head, "distances"):
        distances = head.distances(embedding)
        min_distance = distances.min(dim=1).values
        margin = float(oe_cfg.get("distance_margin", 5.0))
        loss = loss + float(oe_cfg.get("distance_weight", 0.0)) * F.relu(margin - min_distance).pow(2).mean()
    return loss
