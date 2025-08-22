from pathlib import Path
import shutil
import random
from typing import Dict, List, Tuple

from PIL import Image

from papers2code.tools.cifar_adapter import sample_cifar_batches


def _is_img(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg"}


def _has_cifar_batches(root: Path) -> bool:
    return any(root.glob("data_batch_*")) or (root / "test_batch").exists()


def _scan_class_dirs(root: Path) -> Dict[str, List[Path]]:
    cand_roots = []
    for name in ("train", "training", "Train"):
        if (root / name).exists():
            cand_roots.append(root / name)
    for name in ("test", "val", "validation", "Test", "Val"):
        if (root / name).exists():
            cand_roots.append(root / name)
    cand_roots.append(root)  # fallback

    classes: Dict[str, List[Path]] = {}
    seen = set()
    for base in cand_roots:
        for cls_dir in base.iterdir() if base.exists() else []:
            if cls_dir.is_dir():
                images = [p for p in cls_dir.rglob("*") if p.is_file() and _is_img(p)]
                if images:
                    key = f"{base.name}/{cls_dir.name}" if base != root else cls_dir.name
                    if key not in seen:
                        classes[key] = images
                        seen.add(key)
    return classes


def _integrity_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def _sample_from_folders(dataset_dir: Path, out_dir: Path, per_class: int, max_total: int) -> Tuple[Path, Dict[str, int], int]:
    sample_dir = out_dir / "images_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    classes = _scan_class_dirs(dataset_dir)
    class_counts: Dict[str, int] = {}
    broken = 0
    total = 0
    rng = random.Random(42)

    for cls, paths in classes.items():
        good = [p for p in paths if _integrity_ok(p)]
        broken += len(paths) - len(good)
        if not good:
            continue
        rng.shuffle(good)
        take = min(per_class, len(good))
        dest_cls = sample_dir / cls.replace("/", "_")
        dest_cls.mkdir(parents=True, exist_ok=True)
        for p in good[:take]:
            if total >= max_total:
                break
            shutil.copy2(p, dest_cls / p.name)
            total += 1
        class_counts[cls] = take
        if total >= max_total:
            break

    return sample_dir, class_counts, broken


def sample_images_auto(dataset_dir: Path, out_dir: Path, per_class: int = 50, max_total: int = 500) -> Tuple[Path, Dict[str, int], int]:
    """
    Dispatch to CIFAR decoder or folder sampler depending on dataset layout.
    """
    if _has_cifar_batches(dataset_dir):
        return sample_cifar_batches(dataset_dir, out_dir, per_class=per_class, max_total=max_total)
    return _sample_from_folders(dataset_dir, out_dir, per_class=per_class, max_total=max_total)
