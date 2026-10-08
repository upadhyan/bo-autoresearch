import sys
from pathlib import Path

# The benchmark is a script, not a package: its tests import it from benchmark/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
