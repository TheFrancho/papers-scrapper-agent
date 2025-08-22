from pathlib import Path
import pickle
from typing import Dict, List, Tuple

from PIL import Image
import numpy as np


BATCH_GLOB = "data_batch_*"
TEST_BATCH = "test_batch"
META_FILE = "batches.meta"


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")  # CIFAR pickles are py2


def _iter_batches(dataset_dir: Path) -> List[Path]:
    batches = sorted(dataset_dir.glob(BATCH_GLOB))
    tb = dataset_dir / TEST_BATCH
    if tb.exists():
        batches.append(tb)
    return batches


def _load_label_names(dataset_dir: Path) -> List[str] | None:
    meta = dataset_dir / META_FILE
    if not meta.exists():
        # fallback to CIFAR-10 common labels
        return ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"] # HACK Hardcoded for now
    meta_obj = _load_pickle(meta)
    names = meta_obj.get("label_names") or meta_obj.get(b"label_names")
    if isinstance(names, list):
        return [n if isinstance(n, str) else n.decode("utf-8", "ignore") for n in names]
    return None


def _rows_to_image(row: np.ndarray) -> Image.Image:
    r = row[0:1024].reshape(32, 32)
    g = row[1024:2048].reshape(32, 32)
    b = row[2048:3072].reshape(32, 32)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def sample_cifar_batches(
    dataset_dir: Path,
    out_dir: Path,
    per_class: int = 50,
    max_total: int = 500,
) -> Tuple[Path, Dict[str, int], int]:
    sample_dir = out_dir / "images_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    label_names = _load_label_names(dataset_dir)
    per_class_counts: Dict[str, int] = {}
    broken = 0
    total = 0
    per_idx: Dict[int, int] = {}

    for batch_path in _iter_batches(dataset_dir):
        obj = _load_pickle(batch_path)

        # safer get for data
        data = obj.get("data")
        if data is None:
            data = obj.get(b"data")

        labels = obj.get("labels")
        if labels is None:
            labels = obj.get(b"labels")
        if labels is None:  # CIFAR-100 case
            labels = obj.get("fine_labels")
        if labels is None:
            labels = obj.get(b"fine_labels")

        if data is None or labels is None:
            continue

        data = np.asarray(data)
        labels = list(labels)

        for i in range(len(labels)):
            if total >= max_total:
                break
            y = int(labels[i])
            cnt = per_idx.get(y, 0)
            if cnt >= per_class:
                continue
            row = data[i]
            try:
                im = _rows_to_image(row)
            except Exception:
                broken += 1
                continue

            cls = label_names[y] if (label_names and 0 <= y < len(label_names)) else f"class_{y}"
            cls_dir = sample_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            im.save(cls_dir / f"img_{cnt:05d}.png")

            per_idx[y] = cnt + 1
            per_class_counts[cls] = per_class_counts.get(cls, 0) + 1
            total += 1

        if total >= max_total:
            break

    return sample_dir, per_class_counts, broken
