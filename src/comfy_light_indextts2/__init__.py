import sys
from pathlib import Path


source = str(Path(__file__).resolve().parents[2] / 'indexTTS2')
if source not in sys.path:
    sys.path.insert(0, source)
