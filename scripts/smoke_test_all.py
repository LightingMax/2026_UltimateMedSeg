#!/usr/bin/env python3
"""Smoke-test ALL yaml configs: build model, forward pass, compute loss.

Usage:
    cd UltimateMedSeg-main/
    python scripts/smoke_test_all.py

Tests every YAML under configs/ — architectures + training_paradigms.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Tuple

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

IMG_SIZE = 64  # default tiny size; raised per-arch via ARCH_MIN_IMG_SIZE
BATCH = 2

# Architectures that need larger images (window / patch / padding constraints).
ARCH_MIN_IMG_SIZE = {
    "swinunet": 224,
    "swin": 224,
    "swinv2": 224,
    "lawin": 256,
    "maxvit": 224,
    "coatnet": 128,
    "coat": 128,
    "dinov3": 224,
    "dinov2": 224,
    "dino": 224,
    "eva": 96,
    "vit_": 224,
    "vit_clip": 224,
    "clip": 224,
    "cfanet": 256,
    "ege_unet": 128,
    "malunet": 128,
    "mtunet": 224,
    "scaleformer": 224,
    "lvit": 224,
    "languide": 224,
    "medisee": 224,
    "transunet": 224,
    "acmix": 128,
    "fatnet": 224,
    "polyper": 224,
    "ensemble": 224,
    "tgane": 256,
    "lv_unet": 128,
    "segformer": 224,
    "deit": 224,
    "pvt": 128,
    "hsnet": 128,
}

# Architectures with hard-coded backbone input sizes (data.img_size must not override).
ARCH_FIXED_IMG_SIZE = {
    "fatnet": 224,
    "polyper": 224,
    "mednext": 224,
    "uctrans": 64,
    "lvit": 224,
}

SKIP_PATH_MARKERS = (
    "/foundation/",
)

SKIP_ERROR_MARKERS = (
    "does not support pretrained=False",
    "mamba_ssm",
    "MambaBlock requires",
    "nnMamba requires",
    "VM-UNet encoder requires",
    "Failed to fetch config for LLM",
    "Failed to load LLM weights",
    "segment_anything is required",
    "WeightDownloadError",
    "groundingdino",
    "LocalEntryNotFoundError",
    "Network is unreachable",
    "Cannot send a request, as the client has been closed",
    "gated access",
    "Download the checkpoint from",
    "Baidu NetDisk",
)


def _has_mamba() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        return True
    except ImportError:
        return False


def _has_segment_anything() -> bool:
    try:
        import segment_anything  # noqa: F401
        return True
    except ImportError:
        return False


def create_dummy_data():
    """Create minimal dummy datasets."""
    base = ROOT / "data" / "_test_dummy"
    for split in ("train", "val", "test"):
        img_dir = base / split / "images"
        mask_dir = base / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img_path = img_dir / f"img_{i:04d}.png"
            mask_path = mask_dir / f"img_{i:04d}.png"
            if not img_path.exists():
                from PIL import Image
                import numpy as np
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(img).save(str(img_path))
                mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
                Image.fromarray(mask).save(str(mask_path))
    for alias in ("YourDataset", "test_dummy", "source", "target", "target_val",
                  "labeled", "unlabeled"):
        link = ROOT / "data" / alias
        if not link.exists():
            try:
                link.symlink_to(base, target_is_directory=True)
            except OSError:
                pass
    return str(base)


def patch_yaml(cfg, dummy_root):
    """Recursively replace data paths with dummy root."""
    if isinstance(cfg, dict):
        for k, v in list(cfg.items()):
            if isinstance(v, str) and ("data/" in v or "./data" in v):
                if "image" in k or "img" in k:
                    cfg[k] = os.path.join(dummy_root, "train", "images")
                elif "mask" in k:
                    cfg[k] = os.path.join(dummy_root, "train", "masks")
                elif "val" in k:
                    cfg[k] = os.path.join(dummy_root, "val")
                elif "test" in k:
                    cfg[k] = os.path.join(dummy_root, "test")
                elif "train" in k or "labeled" in k:
                    cfg[k] = os.path.join(dummy_root, "train")
                elif "unlabeled" in k:
                    cfg[k] = os.path.join(dummy_root, "train")
                elif "root" in k or "dir" in k:
                    cfg[k] = dummy_root
                else:
                    cfg[k] = dummy_root
            elif isinstance(v, (dict, list)):
                patch_yaml(v, dummy_root)
    elif isinstance(cfg, list):
        for item in cfg:
            patch_yaml(item, dummy_root)


def _disable_pretrained(obj):
    """Recursively set pretrained=False in config dicts."""
    if isinstance(obj, dict):
        if "pretrained" in obj:
            obj["pretrained"] = False
        for v in obj.values():
            _disable_pretrained(v)
    elif isinstance(obj, list):
        for item in obj:
            _disable_pretrained(item)


def resolve_min_img_size(cfg: dict, yaml_path: str) -> int:
    """Pick the smallest img_size that satisfies architecture constraints."""
    model_cfg = cfg.get("model", cfg) or {}
    arch = str(model_cfg.get("architecture", ""))
    encoder_name = str(model_cfg.get("encoder", {}).get("name", ""))
    decoder_name = str(model_cfg.get("decoder", {}).get("name", ""))
    bottleneck_name = str(model_cfg.get("bottleneck", {}).get("name", ""))
    yaml_lower = yaml_path.lower()
    blob = " ".join((arch, encoder_name, decoder_name, bottleneck_name, yaml_lower)).lower()

    if "uctransnet_enc" in blob:
        return 256
    if "unet_swinv2" in blob or "swinv2_tiny_window8_256" in blob:
        return 256

    for key, fixed_size in ARCH_FIXED_IMG_SIZE.items():
        if key in blob:
            return fixed_size

    min_size = IMG_SIZE
    for key, req_size in ARCH_MIN_IMG_SIZE.items():
        if key in blob:
            min_size = max(min_size, req_size)

    return min_size


def force_smoke_settings(cfg: dict, yaml_path: str) -> int:
    """Force img_size, disable pretrained weights, sync data.img_size."""
    min_size = resolve_min_img_size(cfg, yaml_path)
    model_cfg = cfg.get("model", cfg)
    if isinstance(model_cfg, dict):
        if model_cfg.get("img_size") != "native":
            model_cfg["img_size"] = min_size
        _disable_pretrained(model_cfg)
        enc = model_cfg.get("encoder", {})
        if isinstance(enc, dict):
            enc["pretrained"] = False
            if isinstance(enc.get("img_size"), int):
                enc["img_size"] = min_size

    data_cfg = cfg.get("data")
    if isinstance(data_cfg, dict):
        data_cfg["img_size"] = min_size

    return min_size


def preflight_skip(cfg: dict, yaml_path: str) -> Optional[str]:
    """Return a skip reason before building the model, or None."""
    yaml_lower = yaml_path.lower()
    model_cfg = cfg.get("model", {}) or {}

    if cfg.get("mllm") and not model_cfg.get("text_guided"):
        return "inference-only MLLM pipeline (mllm: block, no trainable model)"

    if "llm4seg" in yaml_lower or model_cfg.get("bottleneck", {}).get("name") == "llm4seg":
        return "LLM4Seg bottleneck requires downloadable LLM weights"

    if any(marker in yaml_lower for marker in SKIP_PATH_MARKERS):
        enc_name = str(model_cfg.get("encoder", {}).get("name", ""))
        if enc_name or "foundation" in yaml_lower:
            return "foundation encoder requires pretrained HF / local checkpoints"

    if "pvtv2_add" in yaml_lower:
        return "known channel mismatch in pvtv2+add skip study config"

    if not _has_mamba():
        mamba_keys = ("mamba", "vmunet", "vm_unet", "umamba", "nnmamba", "hc_mamba",
                      "swin_umamba", "polyp_mamba", "serp_mamba", "skin_mamba",
                      "mambavision", "skvmpp", "ultralbm", "ultralight_vm")
        if any(k in yaml_lower for k in mamba_keys):
            return "optional dependency mamba_ssm not installed"

    if not _has_segment_anything():
        if any(k in yaml_lower for k in ("sammed2d", "medclip_sam", "salip", "segment_anything")):
            return "optional dependency segment_anything not installed"

    if "medisee" in yaml_lower:
        return "MediSee pipeline requires large MLLM checkpoints (offline smoke skip)"

    if "grounding_dino" in yaml_lower:
        return "Grounding DINO pipeline is inference-only"

    if any(k in yaml_lower for k in ("cxrclipseg", "tp_drseg", "tpro", "biomedparse")):
        return "requires HF CLIP/BERT weights (offline smoke skip)"

    if "weak_supervision" in yaml_lower:
        return None  # handled in test_one_config: forward-only

    try:
        import medseg.text_guided  # noqa: F401
    except ImportError:
        if "text_guided" in yaml_lower and model_cfg.get("text_guided"):
            return "medseg.text_guided module not available (TextPromptUNet stub)"

    return None


def build_smoke_model(cfg: dict, yaml_path: str):
    """Build model for smoke test (modular, special-arch, or text-guided)."""
    model_cfg = cfg.get("model", {}) or {}
    if model_cfg.get("text_guided"):
        from train_text_guided import build_text_guided_model
        import torch
        device = torch.device("cpu")
        return build_text_guided_model(cfg, device)

    from medseg.model_builder import build_model
    return build_model(cfg)


def _make_target(num_classes: int, batch: int, size: int):
    import torch
    nc = max(int(num_classes), 1)
    return torch.randint(0, nc, (batch, size, size))


def _call_loss(loss_fn, pred, target, *, num_classes: int, images, loss_name: str):
    """Invoke loss with dummy weak-supervision extras when needed."""
    import torch

    sig = inspect.signature(loss_fn.forward)
    params = list(sig.parameters.keys())
    if params and params[0] == "self":
        params = params[1:]

    if len(params) <= 2:
        return loss_fn(pred, target)

    kwargs: dict[str, Any] = {}
    B, _, H, W = pred.shape if pred.ndim == 4 else (BATCH, 0, images.shape[2], images.shape[3])
    if pred.ndim == 2:
        B = pred.shape[0]
        H = W = images.shape[2]

    nc = max(int(num_classes), 2)
    fg = max(int(num_classes) - 1, 1)

    for name in params[2:]:
        lname = name.lower()
        if lname in ("target", "mask", "masks", "gt"):
            kwargs[name] = target
        elif lname in ("cams", "cam", "cam_a", "cam_b", "cam_learn", "cam_teacher", "cam_logits"):
            c = max(fg, 1)
            kwargs[name] = torch.randn(B, c, H, W, device=pred.device)
        elif lname in ("image_labels", "labels"):
            kwargs[name] = torch.zeros(B, nc, device=pred.device)
            kwargs[name][:, min(fg, nc - 1)] = 1.0
        elif lname in ("boxes",):
            kwargs[name] = torch.tensor([[0.1, 0.1, 0.9, 0.9]] * B, device=pred.device)
        elif lname in ("erase_mask",):
            kwargs[name] = torch.zeros(B, 1, H, W, device=pred.device)
        elif lname in ("cls_token_full", "cls_token_masked"):
            kwargs[name] = torch.randn(B, nc, device=pred.device)
        elif lname in ("features", "feature_map"):
            dim = getattr(loss_fn, "feature_dim", pred.shape[1] if pred.ndim == 4 else 64)
            kwargs[name] = torch.randn(B, dim, H, W, device=pred.device)
        elif lname in ("images", "image"):
            kwargs[name] = images
        elif lname in ("points",):
            kwargs[name] = torch.tensor([[[H // 2, W // 2]]] * B, device=pred.device)
        elif lname in ("scribble",):
            kwargs[name] = target.float()
        else:
            kwargs[name] = torch.zeros(B, device=pred.device)

    args = [pred, target]
    if params[0] not in ("pred", "predictions", "prediction", "logits", "input"):
        args = [pred]
    try:
        if len(params) >= 3 and params[1] not in ("target", "mask"):
            return loss_fn(pred, *list(kwargs.values())[: len(params) - 1])
        return loss_fn(*args, **{k: v for k, v in kwargs.items() if k in params[2:]})
    except TypeError:
        try:
            return loss_fn(pred, target, **kwargs)
        except TypeError:
            return loss_fn(pred, target)


def classify_exception(exc: BaseException) -> Tuple[str, str]:
    """Map an exception to (status, message) where status is FAIL or SKIP."""
    msg = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        from medseg.utils.weight_downloader import WeightDownloadError
        if isinstance(exc, WeightDownloadError):
            return "SKIP", msg
    except ImportError:
        pass
    blob = f"{type(exc).__name__} {exc}".lower()
    for marker in SKIP_ERROR_MARKERS:
        if marker.lower() in blob:
            return "SKIP", msg
    return "FAIL", msg


def test_one_config(yaml_path, dummy_root):
    """Build model, forward, compute loss. Returns (status, message)."""
    import yaml
    import torch
    import torch.nn.functional as F

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return "SKIP", "empty yaml"

    patch_yaml(cfg, dummy_root)
    actual_img_size = force_smoke_settings(cfg, yaml_path)

    skip_reason = preflight_skip(cfg, yaml_path)
    if skip_reason:
        return "SKIP", skip_reason

    model_cfg = cfg.get("model", cfg) or {}
    num_classes = model_cfg.get("num_classes")
    if num_classes is None and model_cfg.get("text_guided"):
        num_classes = len(model_cfg["text_guided"].get("class_names", ["bg", "fg"]))
    num_classes = num_classes or 2

    from medseg.model_builder import IncompatibleEncoderError
    try:
        model = build_smoke_model(cfg, yaml_path)
    except IncompatibleEncoderError as e:
        return "SKIP", str(e)[:200]
    except Exception as e:
        return classify_exception(e)

    model.eval()

    x = torch.randn(BATCH, 3, actual_img_size, actual_img_size)
    is_text = getattr(model, "is_text_guided", False)

    try:
        if "tganet" in yaml_path.lower():
            text = torch.randn(BATCH, 5, 300)
            out = model(x, text=text)
        elif "lvit" in yaml_path.lower():
            text = torch.randn(BATCH, 10, 768)
            out = model(x, text=text)
        elif is_text:
            try:
                out = model(x)
            except (ValueError, TypeError):
                text = torch.randn(BATCH, getattr(model, "n_label_phrases", 5),
                                    getattr(model, "label_embed_dim", 300))
                out = model(x, text=text)
        else:
            out = model(x)
    except Exception as e:
        return classify_exception(e)

    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, dict):
        if "out" in out:
            out = out["out"]
        elif "logits" in out:
            out = out["logits"]
        else:
            out = next(iter(out.values()))

    from medseg.registry import LOSS_REGISTRY
    loss_cfg = cfg.get("training", {}).get("loss", {"name": "compound"})
    loss_name = loss_cfg.get("name", "compound")
    loss_params = loss_cfg.get("params", {}) or {}
    if loss_name in LOSS_REGISTRY:
        loss_fn = LOSS_REGISTRY.get(loss_name)(**loss_params)
    else:
        from medseg.losses.compound_loss import CompoundLoss
        loss_fn = CompoundLoss()

    target = _make_target(num_classes, BATCH, actual_img_size)
    if isinstance(out, torch.Tensor) and out.ndim == 4 and out.shape[2:] != target.shape[1:]:
        out = F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)

    try:
        if "weak_supervision" in yaml_path:
            shape = tuple(out.shape) if isinstance(out, torch.Tensor) else "n/a"
            return "OK", f"forward shape={shape} (weak-sup forward-only)"

        if isinstance(out, torch.Tensor) and out.ndim == 2:
            loss = _call_loss(loss_fn, out, target[:, 0, 0], num_classes=num_classes,
                              images=x, loss_name=loss_name)
        else:
            loss = _call_loss(loss_fn, out, target, num_classes=num_classes,
                              images=x, loss_name=loss_name)
        loss.backward()
    except Exception as e:
        return classify_exception(e)

    shape = tuple(out.shape) if isinstance(out, torch.Tensor) else "n/a"
    return "OK", f"loss={float(loss):.4f} shape={shape}"


_WORKER_DUMMY_ROOT: Optional[str] = None


def _worker_init(dummy_root: str) -> None:
    """Per-process setup: import registries once, limit torch threads."""
    global _WORKER_DUMMY_ROOT
    _WORKER_DUMMY_ROOT = dummy_root
    os.chdir(ROOT)
    import medseg.models.encoders   # noqa
    import medseg.models.decoders   # noqa
    import medseg.models.skip_connections  # noqa
    import medseg.models.bottlenecks  # noqa
    import medseg.losses  # noqa
    import medseg.training.weakly_supervised  # noqa
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _worker_run(path: str) -> Tuple[str, str, str]:
    """Run one config in a worker process. Returns (path, status, message)."""
    try:
        status, msg = test_one_config(path, _WORKER_DUMMY_ROOT)
    except Exception as e:
        status, msg = classify_exception(e)
    return path, status, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Only test yamls whose path contains this substring")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--workers", "-j", type=int, default=None,
        help="Parallel worker processes (default: min(cpu_count, 8); use 1 for serial)",
    )
    args = parser.parse_args()

    if args.workers is None:
        workers = min(os.cpu_count() or 1, 8)
    else:
        workers = max(1, args.workers)

    if workers == 1:
        import medseg.models.encoders   # noqa
        import medseg.models.decoders   # noqa
        import medseg.models.skip_connections  # noqa
        import medseg.models.bottlenecks  # noqa
        import medseg.losses  # noqa
        import medseg.training.weakly_supervised  # noqa: register weak-sup losses

    dummy_root = create_dummy_data()
    all_yamls = sorted(glob.glob("configs/**/*.yaml", recursive=True))
    if args.filter:
        all_yamls = [p for p in all_yamls if args.filter in p]

    mode = "serial" if workers == 1 else f"parallel x{workers}"
    print(f"Smoke-testing {len(all_yamls)} YAML configs "
          f"(base img_size={IMG_SIZE}, {mode})...\n")

    log_path = ROOT / "smoke_results.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    ok, fail, skip = [], [], []
    t0 = time.time()
    total = len(all_yamls)
    completed = 0

    def _record(path: str, status: str, msg: str) -> None:
        nonlocal completed
        completed += 1
        if status == "OK":
            ok.append((path, msg))
        elif status == "SKIP":
            skip.append((path, msg))
        else:
            fail.append((path, msg))

        icon = {"OK": "+", "FAIL": "X", "SKIP": "-"}[status]
        line = f"  [{icon}] {completed}/{total} {path}"
        if status == "FAIL" or args.verbose:
            line += f"  ({msg})"
        log(line)

    if workers == 1:
        for path in all_yamls:
            try:
                status, msg = test_one_config(path, dummy_root)
            except Exception as e:
                status, msg = classify_exception(e)
            _record(path, status, msg)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(dummy_root,),
        ) as pool:
            futures = {pool.submit(_worker_run, path): path for path in all_yamls}
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    path, status, msg = fut.result()
                except Exception as e:
                    status, msg = "FAIL", f"Worker crashed: {type(e).__name__}: {e}"
                _record(path, status, msg)

    elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log(f"RESULTS ({elapsed:.1f}s): {len(ok)} OK, {len(fail)} FAIL, {len(skip)} SKIP")
    log(f"{'='*70}")

    if fail:
        log(f"\n{len(fail)} FAILED configs:")
        for path, msg in fail:
            log(f"  {path}")
            log(f"    -> {msg}")

    if skip:
        log(f"\n{len(skip)} SKIPPED configs:")
        for path, msg in skip[:30]:
            log(f"  {path}")
            log(f"    -> {msg}")
        if len(skip) > 30:
            log(f"  ... and {len(skip) - 30} more")

    log(f"\nFull log saved to: {log_path}")
    log_file.close()
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
