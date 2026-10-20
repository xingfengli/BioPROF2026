from __future__ import annotations

from pathlib import Path

import pandas as pd

from .audio import AudioConfig, LogMelFrontend
from .augment import AugmentConfig, BackgroundNoiseMixer, SpecAugment, WaveformAugment, mixup_batch
from .config import ensure_output_dir
from .datasets import AudioManifestDataset, EmbeddingManifestDataset, make_class_balanced_sampler
from .models import SpectrogramClassifier, apply_finetune_mode, backbone_uses_cached_embeddings, model_expects_waveform
from .osr_training import (
    attach_osr_modules,
    auxiliary_osr_loss,
    forward_with_optional_embedding,
    needs_training_embedding,
    oe_loss,
)
from .reproducibility import make_torch_generator, seed_everything, seed_worker


def _collate(batch):
    import torch

    labels = torch.stack([item["label_id"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]
    names = [item["label"] for item in batch]
    is_known = [item["is_known"] for item in batch]
    result = {"label_id": labels, "path": paths, "label": names, "is_known": is_known}
    if "embedding" in batch[0]:
        result["embedding"] = torch.stack([item["embedding"] for item in batch], dim=0)
    else:
        result["waveform"] = torch.stack([item["waveform"] for item in batch], dim=0)
    return result


def run_training(cfg: dict) -> Path:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    seed = int(cfg.get("seed", 26))
    deterministic = bool(cfg.get("training", {}).get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    out = ensure_output_dir(cfg)
    manifest = Path(cfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest

    audio_cfg = AudioConfig(**cfg["audio"])
    aug_cfg = AugmentConfig(**cfg.get("augmentation", {}))
    model_cfg = cfg["model"]
    uses_embeddings = backbone_uses_cached_embeddings(model_cfg["backbone"])
    if uses_embeddings:
        train_ds = EmbeddingManifestDataset(manifest, "train", cfg, known_only=True)
        val_ds = EmbeddingManifestDataset(manifest, "val", cfg, known_only=True)
    else:
        train_ds = AudioManifestDataset(manifest, "train", audio_cfg, known_only=True)
        val_ds = AudioManifestDataset(manifest, "val", audio_cfg, known_only=True)
    oe_cfg = cfg.get("training", {}).get("oe", {})
    oe_enabled = bool(oe_cfg.get("enabled", False))
    oe_ds = None
    if oe_enabled:
        oe_split = str(oe_cfg.get("split", "oe"))
        if uses_embeddings:
            oe_ds = EmbeddingManifestDataset(manifest, oe_split, cfg, unknown_only=True)
        else:
            oe_ds = AudioManifestDataset(manifest, oe_split, audio_cfg, unknown_only=True)
        if len(oe_ds) == 0:
            raise ValueError(f"OE is enabled but no unknown samples were found in split '{oe_split}'.")

    train_generator = make_torch_generator(seed)
    sampler_generator = make_torch_generator(seed + 1)
    val_generator = make_torch_generator(seed + 2)
    sampler = (
        make_class_balanced_sampler(train_ds, generator=sampler_generator)
        if cfg["training"].get("class_balanced_sampler", True)
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=_collate,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=_collate,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    oe_loader = None
    if oe_ds is not None:
        oe_loader = DataLoader(
            oe_ds,
            batch_size=int(oe_cfg.get("batch_size", cfg["training"]["batch_size"])),
            shuffle=True,
            num_workers=cfg["training"].get("num_workers", 4),
            collate_fn=_collate,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(seed + 3),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramClassifier(
        model_cfg["backbone"],
        num_classes=len(cfg["known_labels"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        in_channels=1,
        embedding_dim=getattr(train_ds, "embedding_dim", None),
    ).to(device)
    finetune_summary = apply_finetune_mode(
        model,
        model_cfg["backbone"],
        mode=model_cfg.get("finetune_mode", "full_finetune"),
    )
    attach_osr_modules(model, cfg, len(cfg["known_labels"]))
    model.to(device)
    print(f"Fine-tuning summary: {finetune_summary}")
    frontend = None if uses_embeddings or model_expects_waveform(model) else LogMelFrontend(audio_cfg).to(device)
    waveform_aug = WaveformAugment(aug_cfg)
    spec_aug = SpecAugment(aug_cfg)
    noise_paths = []
    if aug_cfg.background_noise_prob > 0:
        train_df = train_ds.df
        noise_df = train_df[train_df["label"].isin(list(aug_cfg.background_noise_labels))]
        noise_paths = [Path.cwd() / path for path in noise_df["filepath"].tolist()]
    noise_mixer = BackgroundNoiseMixer(aug_cfg, noise_paths, audio_cfg)

    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable model parameters were found.")
    if cfg["training"].get("optimizer", "adamw").lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=cfg["training"]["learning_rate"],
            momentum=0.9,
            weight_decay=cfg["training"].get("weight_decay", 0.0),
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"].get("weight_decay", 0.01),
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    best_acc = -1.0
    history = []
    need_embedding = needs_training_embedding(cfg)
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        oe_iter = iter(oe_loader) if oe_loader is not None else None
        for batch in train_loader:
            target = batch["label_id"].to(device)
            if uses_embeddings:
                inputs = batch["embedding"].to(device)
            else:
                waveform = batch["waveform"].to(device)
                waveform = waveform_aug(waveform)
                waveform = noise_mixer(waveform)
                inputs = _prepare_model_inputs(model, frontend, waveform, spec_aug=spec_aug)
            inputs, y_a, y_b, lam = mixup_batch(inputs, target, float(aug_cfg.mixup_alpha))

            optimizer.zero_grad(set_to_none=True)
            logits, embedding = forward_with_optional_embedding(model, inputs, need_embedding)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            extra_loss = auxiliary_osr_loss(model, embedding, target, cfg)
            if extra_loss is not None:
                loss = loss + extra_loss
            if oe_iter is not None:
                try:
                    oe_batch = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(oe_loader)
                    oe_batch = next(oe_iter)
                if uses_embeddings:
                    oe_inputs = oe_batch["embedding"].to(device)
                else:
                    oe_waveform = oe_batch["waveform"].to(device)
                    oe_waveform = waveform_aug(oe_waveform)
                    oe_waveform = noise_mixer(oe_waveform)
                    oe_inputs = _prepare_model_inputs(model, frontend, oe_waveform, spec_aug=spec_aug)
                oe_logits, oe_embedding = forward_with_optional_embedding(model, oe_inputs, need_embedding)
                loss = loss + oe_loss(model, oe_logits, oe_embedding, cfg)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * target.size(0)
            total += target.size(0)
            correct += int((logits.argmax(dim=1) == target).sum().item())
        scheduler.step()

        val_acc = _closed_set_accuracy(model, frontend, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "train_acc": correct / max(1, total),
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(row)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "label_names": cfg["known_labels"],
                    "finetune_summary": finetune_summary,
                    "seed": seed,
                    "deterministic": deterministic,
                },
                out / "checkpoint_best.pt",
            )

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    return out / "checkpoint_best.pt"


def _prepare_model_inputs(model, frontend, waveform, spec_aug=None):
    if model_expects_waveform(model):
        return waveform
    if frontend is None:
        raise RuntimeError("A spectrogram frontend is required for this model.")
    spec = frontend(waveform).to(waveform.device)
    if spec_aug is not None:
        spec = spec_aug(spec)
    return spec


def _closed_set_accuracy(model, frontend, loader, device) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["label_id"].to(device)
            if "embedding" in batch:
                inputs = batch["embedding"].to(device)
            else:
                waveform = batch["waveform"].to(device)
                inputs = _prepare_model_inputs(model, frontend, waveform)
            logits = model(inputs)
            correct += int((logits.argmax(dim=1) == target).sum().item())
            total += target.size(0)
    return correct / max(1, total)
