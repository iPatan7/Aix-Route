"""Make ``integrations`` importable when pytest runs from anywhere."""

from __future__ import annotations

import sys
from pathlib import Path

# integrations/tests/ -> integrations/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
