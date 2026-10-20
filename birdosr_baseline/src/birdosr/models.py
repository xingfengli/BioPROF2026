from __future__ import annotations

from collections.abc import Mapping


VALID_FINETUNE_MODES = {"linear_probe", "partial_finetune", "full_finetune"}
BIRDMAE_BASE_MODEL_ID = "DBD-research-group/Bird-MAE-Base"
OPENSOUNDSCAPE_DEFAULT_ARCHITECTURE = "resnet18"


def _patch_first_conv(module, in_channels: int):
    import torch
    import torch.nn as nn

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.in_channels == in_channels:
                return
            new_conv = nn.Conv2d(
                in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            with torch.no_grad():
                if child.weight.shape[1] == 3 and in_channels == 1:
                    new_conv.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if child.bias is not None:
                    new_conv.bias.copy_(child.bias)
            setattr(module, name, new_conv)
            return
        _patch_first_conv(child, in_channels)


def _set_requires_grad(module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _module_by_dotted_name(module, dotted_name: str):
    current = module
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _last_item(module):
    try:
        if len(module) > 0:
            return module[-1]
    except TypeError:
        return None
    return None


def _dedupe_modules(modules: list) -> list:
    unique = []
    seen = set()
    for module in modules:
        if id(module) not in seen:
            unique.append(module)
            seen.add(id(module))
    return unique


def _head_modules(model) -> list:
    heads = []
    for name in ("fc", "classifier", "head", "heads", "last_linear"):
        module = getattr(model, name, None)
        if module is not None:
            heads.append(module)
    if hasattr(model, "get_classifier"):
        try:
            module = model.get_classifier()
            if module is not None:
                heads.append(module)
        except Exception:
            pass
    return _dedupe_modules(heads)


def _prepend_classifier_dropout(model, dropout: float):
    import torch.nn as nn

    if dropout <= 0:
        return model

    for name in ("fc", "classifier", "head", "heads", "last_linear"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Linear):
            setattr(model, name, nn.Sequential(nn.Dropout(dropout), module))
            return model
        if isinstance(module, nn.Sequential):
            for index in range(len(module) - 1, -1, -1):
                if isinstance(module[index], nn.Linear):
                    module[index] = nn.Sequential(nn.Dropout(dropout), module[index])
                    return model
    raise ValueError("Could not locate a linear classifier head for dropout injection.")


def _last_stage_modules(model, backbone: str) -> list:
    stages = []
    for name in ("layer4", "stage4", "final_stage"):
        module = getattr(model, name, None)
        if module is not None:
            stages.append(module)

    for name in ("stages", "blocks", "layers", "features"):
        module = getattr(model, name, None)
        last = _last_item(module) if module is not None else None
        if last is not None:
            stages.append(last)

    # Common HuggingFace layouts.
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        for dotted_name in (
            "audio_spectrogram_transformer.encoder.layer",
            "audio_spectrogram_transformer.encoder.layers",
            "audio_spectrogram_transformer.encoder.blocks",
            "encoder.layer",
            "encoder.layers",
            "encoder.blocks",
            "layer",
            "layers",
            "blocks",
            "model.layers",
            "model.blocks",
        ):
            module = _module_by_dotted_name(encoder, dotted_name)
            last = _last_item(module) if module is not None else None
            if last is not None:
                stages.append(last)

        for name, module in encoder.named_modules():
            if name.endswith(("layer", "layers", "blocks", "stages")):
                last = _last_item(module)
                if last is not None:
                    stages.append(last)

    # Torchvision EfficientNet has the most semantic final MBConv block at
    # features[-2], followed by a final convolution at features[-1].
    if backbone == "efficientnet_b0" and hasattr(model, "features"):
        try:
            if len(model.features) > 1:
                stages.append(model.features[-2])
        except TypeError:
            pass

    return _dedupe_modules(stages)


def apply_finetune_mode(model, backbone: str, mode: str = "full_finetune") -> dict[str, int | str]:
    """Apply one of the supported fine-tuning modes.

    Modes:
    - linear_probe: freeze the feature extractor and train classifier heads only.
    - partial_finetune: train classifier heads plus the final backbone stage.
    - full_finetune: train every parameter.
    """
    if mode not in VALID_FINETUNE_MODES:
        valid = ", ".join(sorted(VALID_FINETUNE_MODES))
        raise ValueError(f"Unsupported finetune_mode '{mode}'. Valid modes: {valid}")
    if backbone_uses_cached_embeddings(backbone) and mode != "linear_probe":
        raise ValueError("BMZ embedding backbones use frozen cached embeddings and require finetune_mode='linear_probe'.")

    if mode == "full_finetune":
        _set_requires_grad(model, True)
    else:
        _set_requires_grad(model, False)
        head_modules = _head_modules(model)
        if not head_modules:
            raise ValueError("Could not locate classifier head for fine-tuning.")
        for module in head_modules:
            _set_requires_grad(module, True)
        if mode == "partial_finetune":
            stage_modules = _last_stage_modules(model, backbone)
            if not stage_modules:
                raise ValueError(f"Could not locate final stage for partial fine-tuning of backbone '{backbone}'.")
            for module in stage_modules:
                _set_requires_grad(module, True)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "finetune_mode": mode,
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


class SpectrogramClassifier:
    def __new__(
        cls,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        in_channels: int = 1,
        embedding_dim: int | None = None,
    ):
        import torch.nn as nn

        if backbone.startswith("bmz:"):
            if embedding_dim is None:
                raise ValueError("embedding_dim is required for a bmz:* embedding backbone.")
            return build_embedding_classifier(embedding_dim, num_classes, dropout)

        if backbone == "birdmae":
            return build_birdmae_classifier(BIRDMAE_BASE_MODEL_ID, num_classes, pretrained, dropout)

        if backbone.startswith("birdmae:"):
            model_id = backbone.split(":", 1)[1] or BIRDMAE_BASE_MODEL_ID
            return build_birdmae_classifier(model_id, num_classes, pretrained, dropout)

        if backbone == "opensoundscape":
            return build_opensoundscape_classifier(
                OPENSOUNDSCAPE_DEFAULT_ARCHITECTURE,
                num_classes,
                pretrained,
                dropout,
                in_channels,
            )

        if backbone.startswith("opensoundscape:"):
            architecture = backbone.split(":", 1)[1] or OPENSOUNDSCAPE_DEFAULT_ARCHITECTURE
            return build_opensoundscape_classifier(architecture, num_classes, pretrained, dropout, in_channels)

        if backbone.startswith("timm:"):
            import timm

            model_name = backbone.split(":", 1)[1]
            return timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=num_classes,
                drop_rate=dropout,
            )

        if backbone.startswith("hf:"):
            return build_huggingface_audio_classifier(backbone.split(":", 1)[1], num_classes, dropout)

        from torchvision import models

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            _patch_first_conv(model, in_channels)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            return model

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            _patch_first_conv(model, in_channels)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            return model

        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            _patch_first_conv(model, in_channels)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            return model

        raise ValueError(f"Unsupported backbone: {backbone}")


def model_expects_waveform(model) -> bool:
    return bool(getattr(model, "expects_waveform", False))


def backbone_uses_cached_embeddings(backbone: str) -> bool:
    return backbone.startswith("bmz:")


def build_embedding_classifier(embedding_dim: int, num_classes: int, dropout: float):
    import torch.nn as nn

    class _EmbeddingClassifier(nn.Module):
        uses_cached_embeddings = True

        def __init__(self):
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embedding_dim, num_classes))

        def forward(self, embedding):
            return self.head(embedding)

        def extract_embedding(self, embedding):
            return embedding

        def classify_embedding(self, embedding):
            return self.head(embedding)

    return _EmbeddingClassifier()


def build_opensoundscape_classifier(
    architecture: str,
    num_classes: int,
    pretrained: bool,
    dropout: float,
    in_channels: int,
):
    import inspect

    try:
        from opensoundscape.ml import cnn_architectures
    except ImportError as exc:
        raise RuntimeError(
            "OpenSoundscape is required for an opensoundscape:* backbone. "
            "Install the project requirements before training."
        ) from exc

    factory = getattr(cnn_architectures, architecture, None)
    if factory is None or not callable(factory):
        available = sorted(
            name
            for name in dir(cnn_architectures)
            if not name.startswith("_") and callable(getattr(cnn_architectures, name))
        )
        raise ValueError(
            f"Unsupported OpenSoundscape architecture '{architecture}'. "
            f"Available callables include: {', '.join(available)}"
        )

    parameters = inspect.signature(factory).parameters
    kwargs = {}
    if "num_classes" in parameters:
        kwargs["num_classes"] = num_classes
    else:
        raise ValueError(f"OpenSoundscape architecture '{architecture}' does not accept num_classes.")
    if "freeze_feature_extractor" in parameters:
        kwargs["freeze_feature_extractor"] = False
    if "weights" in parameters:
        kwargs["weights"] = "DEFAULT" if pretrained else None
    if "num_channels" in parameters:
        kwargs["num_channels"] = in_channels

    model = factory(**kwargs)
    _patch_first_conv(model, in_channels)
    return _prepend_classifier_dropout(model, dropout)


def _infer_hidden_size(config) -> int:
    for name in ("hidden_size", "d_model", "embed_dim", "encoder_embed_dim", "projection_dim"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 768


def _pool_encoder_outputs(outputs):
    import torch

    def _pool_tensor(tensor):
        if tensor.dim() == 3:
            return tensor.mean(dim=1)
        return tensor

    if torch.is_tensor(outputs):
        return _pool_tensor(outputs)

    if isinstance(outputs, Mapping):
        for key in ("pooler_output", "embedding", "embeddings"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return _pool_tensor(value)
        value = outputs.get("last_hidden_state")
        if torch.is_tensor(value):
            return _pool_tensor(value)

    for name in ("pooler_output", "embedding", "embeddings"):
        value = getattr(outputs, name, None)
        if torch.is_tensor(value):
            return _pool_tensor(value)
    value = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(value):
        return _pool_tensor(value)

    if isinstance(outputs, (tuple, list)) and outputs:
        return _pool_encoder_outputs(outputs[0])

    raise ValueError("Could not pool encoder outputs into a batch embedding.")


def _move_feature_batch_to_device(batch, device):
    import torch

    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(value.to(device) if torch.is_tensor(value) else value for value in batch)
    return batch


def _stack_feature_items(items):
    import torch

    first = items[0]
    if torch.is_tensor(first):
        items = [item.squeeze(0) if item.dim() > 0 and item.shape[0] == 1 else item for item in items]
        return torch.stack(items, dim=0)
    if isinstance(first, Mapping):
        stacked = {}
        for key in first.keys():
            values = [item[key] for item in items]
            if all(torch.is_tensor(value) for value in values):
                values = [value.squeeze(0) if value.dim() > 0 and value.shape[0] == 1 else value for value in values]
                stacked[key] = torch.stack(values, dim=0)
            else:
                stacked[key] = values
        return stacked
    raise TypeError(f"Unsupported feature extractor output type: {type(first)!r}")


def build_birdmae_classifier(model_id: str, num_classes: int, pretrained: bool, dropout: float):
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

    class _BirdMAEClassifier(nn.Module):
        expects_waveform = True

        def __init__(self):
            super().__init__()
            self.model_id = model_id
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
            self.sample_rate = int(getattr(self.feature_extractor, "sampling_rate", 32_000) or 32_000)
            if pretrained:
                self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            else:
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                self.encoder = AutoModel.from_config(config, trust_remote_code=True)
            hidden = _infer_hidden_size(self.encoder.config)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

        def _extract_features(self, waveform):
            waveform = waveform.detach().float()
            if waveform.dim() == 3 and waveform.shape[1] == 1:
                waveform = waveform[:, 0, :]
            elif waveform.dim() != 2:
                raise ValueError(f"BirdMAE expects waveform with shape [B, T] or [B, 1, T], got {tuple(waveform.shape)}")

            audio = [item for item in waveform.cpu().numpy()]
            features = self._call_feature_extractor(audio)
            return _move_feature_batch_to_device(features, waveform.device)

        def _call_feature_extractor(self, audio):
            try:
                return self.feature_extractor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
            except TypeError:
                try:
                    return self.feature_extractor(audio, return_tensors="pt")
                except TypeError:
                    pass
            except Exception:
                pass

            items = []
            for item in audio:
                try:
                    items.append(self.feature_extractor(item, sampling_rate=self.sample_rate, return_tensors="pt"))
                except TypeError:
                    try:
                        items.append(self.feature_extractor(item, return_tensors="pt"))
                    except TypeError:
                        items.append(self.feature_extractor(item))
            return _stack_feature_items(items)

        def _encode(self, features):
            if isinstance(features, Mapping):
                try:
                    return self.encoder(**features)
                except TypeError:
                    for key in ("input_values", "input_features", "pixel_values", "features"):
                        value = features.get(key)
                        if torch.is_tensor(value):
                            return self.encoder(value)
                    for value in features.values():
                        if torch.is_tensor(value):
                            return self.encoder(value)
                    raise
            return self.encoder(features)

        def extract_embedding(self, waveform):
            features = self._extract_features(waveform)
            outputs = self._encode(features)
            return _pool_encoder_outputs(outputs)

        def classify_embedding(self, embedding):
            return self.head(embedding)

        def forward(self, waveform):
            pooled = self.extract_embedding(waveform)
            return self.classify_embedding(pooled)

    return _BirdMAEClassifier()


def build_huggingface_audio_classifier(model_id: str, num_classes: int, dropout: float):
    import torch.nn as nn
    from transformers import AutoModel

    class _HuggingFaceAudioClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            hidden = getattr(self.encoder.config, "hidden_size", None)
            if hidden is None:
                raise ValueError("Could not infer hidden_size from HuggingFace encoder config.")
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

        def forward(self, x):
            outputs = self.encoder(x)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state.mean(dim=1)
            return self.head(pooled)

    return _HuggingFaceAudioClassifier()
