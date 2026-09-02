"""Put the repo root on ``sys.path`` for the test session.

``src/mjlab_microduck`` is installed into the venv, but ``qd/`` deliberately is
not part of the distribution (see ``qd/README.md``) — it is imported from the
working tree. pytest's default "prepend" import mode already inserts the
directory holding the root ``conftest.py``; the explicit insert below keeps that
working regardless of import mode or invocation directory.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
