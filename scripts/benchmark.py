import json
import time
from pathlib import Path

import numpy as np
import torch
from timestral import TimestralForecaster

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
manifest = json.loads((ROOT / "data/manifest.json").read_text())
model = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-209M", revision=manifest["model_revision"], local_files_only=True
)
print("Parameters", sum(p.numel() for p in model.model.parameters()), flush=True)
for context, batch in [(100, 16), (250, 16), (500, 16), (250, 32)]:
    x = np.random.default_rng(42).normal(0, .01, (batch, context)).astype("float32")
    t0 = time.perf_counter()
    y = model.forecast(x, prediction_length=20, clip_nonnegative=False)
    print(context, batch, "seconds", time.perf_counter()-t0, "shape", y.quantiles.shape,
          "finite", bool(torch.isfinite(y.quantiles).all()), flush=True)
