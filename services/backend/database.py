from pathlib import Path
from sqlmodel import create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "siamesescribe.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(
    f"sqlite:///{DB_PATH.resolve().as_posix()}",
    connect_args={"check_same_thread": False}
)