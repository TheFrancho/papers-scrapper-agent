import os
from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    artifacts_dir: Path = Path(os.getenv("P2C_ARTIFACTS_DIR", "./artifacts"))
    samples_max_rows: int = int(os.getenv("P2C_SAMPLES_MAX_ROWS", "50000"))
    image_sample_max: int = int(os.getenv("P2C_IMAGE_SAMPLE_MAX", "300"))

settings = Settings()
