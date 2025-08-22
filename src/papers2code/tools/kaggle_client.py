from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


_api: KaggleApi | None = None


def _api_client() -> KaggleApi:
    global _api
    if _api is None:
        _api = KaggleApi()
        _api.authenticate()
    return _api


def kaggle_search_datasets(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    api = _api_client()
    results: List[Dict[str, Any]] = []
    listed = api.dataset_list(search=query)
    for ds in listed[:limit]:
        ref = getattr(ds, "ref", None)
        title = getattr(ds, "title", None)
        size = getattr(ds, "size", None)
        url = f"https://www.kaggle.com/datasets/{ref}" if ref else None
        license_name = None
        try:
            view = api.dataset_view(ref)
            if getattr(view, "licenseName", None):
                license_name = view.licenseName
        except Exception:
            pass
        results.append({
            "ref": ref, "title": title, "size": size, "url": url, "license": license_name
        })
    return results


def kaggle_files_and_size(ref: str) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """
    Return (files, total_size_mb) without downloading
    files = [{name, totalBytes, type}], total_size_mb is float or None
    """
    api = _api_client()
    try:
        lf = api.dataset_list_files(ref)
        files = []
        total = 0
        for f in getattr(lf, "files", []) or []:
            sz = getattr(f, "totalBytes", 0) or 0
            total += sz
            files.append({"name": getattr(f, "name", None), "totalBytes": sz, "type": getattr(f, "type", None)})
        mb = round(total / (1024 * 1024), 3)
        return files, mb
    except Exception:
        return [], None


def kaggle_download_dataset(ref: str, dest: Path) -> None:
    api = _api_client()
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=False)