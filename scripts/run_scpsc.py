"""Command-line entry and callable worker for a SCPSC task."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from scpsc.application import worker
from scpsc.cli import configure_threads, launch, main
from scpsc.run_state import Budget

if __name__ == "__main__":
    raise SystemExit(main())
