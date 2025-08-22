from pathlib import Path
from typing import Dict

import imagehash
from PIL import Image


def profile_images(sample_dir: Path) -> Dict:
    """
    Simple stats over sampled images: total, per-class counts, phash dup rate (approx)
    Expects layout: sample_dir/<class>/*.png
    """
    classes = [p for p in sample_dir.iterdir() if p.is_dir()]
    per_class = {c.name: len([f for f in c.rglob("*") if f.is_file()]) for c in classes}
    total = sum(per_class.values())

    # approximate duplicate rate via perceptual hash buckets
    hashes = {}
    dups = 0
    for c in classes:
        for f in c.rglob("*"):
            if f.is_file():
                try:
                    with Image.open(f) as im:
                        h = str(imagehash.phash(im))
                    hashes.setdefault(h, 0)
                    hashes[h] += 1
                except Exception:
                    pass
    for k, cnt in hashes.items():
        if cnt > 1:
            dups += (cnt - 1)
    dup_rate = (dups / total) if total > 0 else 0.0

    return {
        "modality": "images",
        "total_images": total,
        "per_class": per_class,
        "approx_duplicate_rate": float(dup_rate),
    }
