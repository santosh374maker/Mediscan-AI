import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force a clean, isolated test environment before any src module is imported.
os.environ.setdefault("ENVIRONMENT", "development")
