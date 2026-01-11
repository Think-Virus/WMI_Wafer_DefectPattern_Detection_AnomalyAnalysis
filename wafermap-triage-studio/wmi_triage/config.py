from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    artifacts: Path = root / "artifacts"
    models: Path = artifacts / "models"
    emb_db: Path = artifacts / "emb_db"
    reports: Path = artifacts / "reports"
    cases: Path = artifacts / "cases"


@dataclass(frozen=True)
class RunConfig:
    resize: int = 64
    input_mode: str = "polar4"
    upsample_mode: str = "nearest"
    device: str = "cpu"  # scripts에서 자동으로 덮어쓸 것
