from typing import List, Dict, Literal
Modality = Literal["tabular", "images", "text", "timeseries", "unknown"]

def guess_modality(files: List[Dict]) -> Modality:
    names = [ (f.get("name") or "").lower() for f in files ]
    if any(n.endswith((".csv", ".parquet")) for n in names):
        return "tabular"

    # CIFAR mirrors often expose batch files OR class folders
    if any(n.startswith("data_batch_") or n == "test_batch" for n in names):
        return "images"
    if any(n.endswith((".png", ".jpg", ".jpeg")) for n in names):
        return "images"
    if any("train/" in n or "test/" in n for n in names):
        return "images"
    return "unknown"
