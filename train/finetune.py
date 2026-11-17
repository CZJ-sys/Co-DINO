"""Single-PDE forecasting with joint Encoder-Decoder optimization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.runtime import main

if __name__ == "__main__":
    main("finetune")
