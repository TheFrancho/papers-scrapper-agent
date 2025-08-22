from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class KaggleMeta(BaseModel):
    slug: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    license: Optional[str] = None
    size_mb: Optional[float] = None
    files: List[Dict[str, Any]] = []

class PipelineState(BaseModel):
    paper_source: str
    paper_text: str = ""
    sections: Dict[str, str] = {}
    dataset_candidates: List[Dict[str, Any]] = []
    kaggle_choice: Optional[KaggleMeta] = None
    sample_dir: Optional[str] = None
    dataset_profile: Dict[str, Any] = {}
    method_spec: Dict[str, Any] = {}
    code_scaffold: Dict[str, str] = {}
    wiki_md: str = ""
    issues: List[str] = []
