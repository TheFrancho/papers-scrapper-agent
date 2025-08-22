from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import random
import matplotlib.pyplot as plt

def save_class_bar_chart(per_class: Dict[str, int], out_path: Path) -> None:
    """
    Bar chart with counts per class. Sorted by class name, value labels on bars,
    light grid, larger figure, tight layout. No explicit colors.
    """
    if not per_class:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return

    labels = sorted(per_class.keys())
    counts = [per_class[k] for k in labels]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, counts)
    plt.title("Images per class (sample)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right")

    # Add value labels on top of bars
    for b, val in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.01,
                 f"{val}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_sample_grid(sample_dir: Path, out_path: Path, grid: int = 3) -> None:
    """
    Show a grid of images with round-robin class selection so multiple classes appear.
    Adds small titles per cell and a super-title. No explicit colors/styles.
    """
    # Collect images per class folder
    class_dirs = [p for p in sample_dir.iterdir() if p.is_dir()]
    class_dirs.sort(key=lambda p: p.name.lower())

    per_class_imgs: List[List[Path]] = []
    for c in class_dirs:
        imgs = [p for p in c.glob("*") if p.is_file()]
        # deterministic shuffle so the grid changes little run-to-run
        rng = random.Random(42)
        rng.shuffle(imgs)
        if imgs:
            per_class_imgs.append(imgs)

    N = grid * grid
    picked: List[Tuple[str, Path]] = []
    # Round-robin: cycle across classes taking one at a time until we fill N or run out
    idxs = [0] * len(per_class_imgs)
    k = 0
    while len(picked) < N and per_class_imgs:
        cls_idx = k % len(per_class_imgs)
        imgs = per_class_imgs[cls_idx]
        i = idxs[cls_idx]
        if i < len(imgs):
            picked.append((class_dirs[cls_idx].name, imgs[i]))
            idxs[cls_idx] += 1
        k += 1
        # break if we looped through all classes and couldn’t add more
        if k > len(per_class_imgs) * (max(len(x) for x in per_class_imgs) if per_class_imgs else 0):
            break

    if not picked:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return

    plt.figure(figsize=(grid * 3.2, grid * 3.2))
    for i, (cls, path) in enumerate(picked[:N]):
        plt.subplot(grid, grid, i + 1)
        try:
            with Image.open(path) as im:
                plt.imshow(im)
        except Exception:
            # show empty tile if unreadable
            plt.imshow([[0]])
        plt.axis("off")
        plt.title(cls[:18], fontsize=9, pad=2)

    plt.suptitle("Sample images (round-robin across classes)", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
