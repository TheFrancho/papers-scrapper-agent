from pathlib import Path
from typing import Dict, Any, List
import json

from papers2code.llm.openai_client import chat_json


SCHEMA = {
    "dataset": {"name": str, "num_classes": int, "input_size": [int, int, int]},
    "preprocess": {"normalize": {"mean": [float], "std": [float]},
                   "augment": {"random_crop": bool, "padding": int, "random_flip": bool, "cutout": bool}},
    "model": {"family": str, "depth": int, "widen_factor": int, "dropout": float},
    "train": {"epochs": int, "batch_size": int, "optimizer": str, "lr": float,
              "momentum": float, "weight_decay": float, "scheduler": str},
    "citations": [{"section": str, "quote": str}]
}


def _fallback_for_cifar10() -> Dict[str, Any]:
    # Sensible WRN defaults when the paper doesn’t fully specify
    return {
        "dataset": {"name": "CIFAR-10", "num_classes": 10, "input_size": [3, 32, 32]},
        "preprocess": {
            "normalize": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
            "augment": {"random_crop": True, "padding": 4, "random_flip": True, "cutout": False},
        },
        "model": {"family": "wide_resnet", "depth": 28, "widen_factor": 10, "dropout": 0.3},
        "train": {"epochs": 200, "batch_size": 128, "optimizer": "sgd",
                  "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "scheduler": "cosine"},
        "citations": [],
    }


def extract_methods(paper_text: str, sections: Dict[str, Any], log_dir: Path) -> Dict[str, Any]:
    """
    Uses an LLM to extract methods config from the paper
    Logs prompt/response. Falls back to CIFAR-10 WRN defaults if extraction is incomplete
    """
    excerpt = paper_text[:100_000]
    prompt = (
        "From the research paper excerpt below, extract an implementation plan for the Methods section.\n"
        "Return a strict JSON object with keys:\n"
        "Add 5-10 concise citations as {section, quote}. "
        "Prefer Implementation/Experiments sections; include config-like snippets.\n\n"
        f"Paper excerpt:\n{excerpt}"
        + json.dumps(list(SCHEMA.keys())) + "\n"
        "Where:\n"
        "- dataset: {name, num_classes, input_size [C,H,W]}\n"
        "- preprocess.normalize: {mean, std} (3 floats each for RGB)\n"
        "- preprocess.augment: {random_crop, padding, random_flip, cutout}\n"
        "- model: {family (e.g., 'wide_resnet'), depth, widen_factor, dropout}\n"
        "- train: {epochs, batch_size, optimizer, lr, momentum, weight_decay, scheduler}\n"
        "- citations: array of {section: short label, quote: short supporting snippet}\n"
        "If the paper omits a field, infer reasonable defaults for CIFAR-10/Wide-ResNet and mark that field anyway.\n"
        "Be concise; numeric values should be scalars.\n\n"
        f"Paper excerpt:\n{excerpt}"
    )
    data = chat_json(prompt, log_dir=log_dir, log_name="methods")

    # Minimal validation + fallback completion
    spec = _fallback_for_cifar10()
    if isinstance(data, dict):
        # deep-merge available keys
        for k in ("dataset", "preprocess", "model", "train", "citations"):
            if k in data and data[k]:
                spec[k] = data[k]

    # Coerce shapes / ensure basics
    spec["dataset"].setdefault("name", "CIFAR-10")
    spec["dataset"].setdefault("num_classes", 10)
    spec["dataset"].setdefault("input_size", [3, 32, 32])

    # Defaults for preprocess
    norm = spec.setdefault("preprocess", {}).setdefault("normalize", {})
    norm.setdefault("mean", [0.4914, 0.4822, 0.4465])
    norm.setdefault("std",  [0.2470, 0.2435, 0.2616])
    aug = spec["preprocess"].setdefault("augment", {})
    aug.setdefault("random_crop", True)
    aug.setdefault("padding", 4)
    aug.setdefault("random_flip", True)
    aug.setdefault("cutout", False)

    # Model defaults
    mdl = spec.setdefault("model", {})
    mdl.setdefault("family", "wide_resnet")
    mdl.setdefault("depth", 28)
    mdl.setdefault("widen_factor", 10)
    mdl.setdefault("dropout", 0.3)

    # Training defaults
    tr = spec.setdefault("train", {})
    tr.setdefault("epochs", 200)
    tr.setdefault("batch_size", 128)
    tr.setdefault("optimizer", "sgd")
    tr.setdefault("lr", 0.1)
    tr.setdefault("momentum", 0.9)
    tr.setdefault("weight_decay", 5e-4)
    tr.setdefault("scheduler", "cosine")

    # Save for inspection
    (log_dir / "method_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec
