import json
import time

import numpy as np
import torch
from timestral import TimestralForecaster

from infer import ROOT, load_prices
from vulkan_backend import VulkanEngine, offload

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
rng = np.random.default_rng(42)
engine = VulkanEngine()
for m, k, n in [(7, 35, 19), (256, 768, 2304), (576, 768, 6144), (576, 3072, 768)]:
    layer = torch.nn.Linear(k, n, bias=True)
    x = torch.from_numpy(rng.normal(size=(m, k)).astype("float32"))
    index = engine.add(layer)
    ref = layer(x).detach().numpy()
    t0 = time.perf_counter()
    pred = engine.run(index, x, n).numpy()
    print("Linear", m, k, n, "GPU seconds", time.perf_counter()-t0,
          "max abs error", float(abs(ref-pred).max()), flush=True)
    np.testing.assert_allclose(pred, ref, atol=2e-5, rtol=2e-4)
engine.close()

manifest = json.loads((ROOT / "data/manifest.json").read_text())
model = TimestralForecaster.from_pretrained("mistralai/Timestral-209M",
    revision=manifest["model_revision"], local_files_only=True)
inputs, references = {}, {}
for context in [100, 250, 500]:
    for target in ["returns", "prices"]:
        x = []
        for ticker in ["SPY", "RELIANCE.NS", "EURUSD=X"]:
            p = load_prices(ticker).to_numpy()
            series = np.diff(np.log(p))*100 if target == "returns" else p
            for lag in [1, 100]:
                end = len(series)-lag
                x.append(series[end-context+1:end+1])
        x = np.asarray(x, dtype=np.float32)
        inputs[context, target] = x
        references[context, target] = model.forecast(x, 20,
            clip_nonnegative=(target == "prices")).quantiles.numpy()
        print("Reference", context, target, "done", flush=True)
engine = offload(model.model)
reports = []
for key, x in inputs.items():
    context, target = key
    t0 = time.perf_counter()
    pred = model.forecast(x, 20, clip_nonnegative=(target == "prices")).quantiles.numpy()
    elapsed = time.perf_counter()-t0
    ref = references[key]
    widths = np.maximum(ref[..., 18]-ref[..., 2], 1e-8)
    relative = abs(pred-ref)/widths[..., None]
    record = {
        "context": context, "target": target, "origins": len(x), "elapsed_seconds": elapsed,
        "max_absolute_difference": float(abs(pred-ref).max()),
        "max_difference_divided_80_width": float(relative.max()),
        "median_difference_divided_80_width": float(np.median(relative)),
        "median_sign_agreement": float(np.mean(np.sign(pred[..., 10]) == np.sign(ref[..., 10]))),
    }
    print(json.dumps(record), flush=True)
    assert relative.max() < .001, record
    reports.append(record)
for context in [100, 250, 500]:
    x = rng.normal(0, 1, size=(64, context)).astype("float32")
    t0 = time.perf_counter()
    model.forecast(x, 20, clip_nonnegative=False)
    print("THROUGHPUT", context, "batch64", time.perf_counter()-t0, flush=True)
(ROOT / "results/vulkan_validation.json").write_text(json.dumps(reports, indent=2))
engine.close()
print("Vulkan validation passed", flush=True)
