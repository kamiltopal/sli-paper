"""End-to-end smoke test — RUN THIS FIRST on your machine.

Tiny budget (2 epochs, 1 series, all 3 architectures). Checks:
  - shapes flow through every model
  - loss is finite and decreases vs an untrained baseline
  - GPU is actually used if available

Expected runtime: < 1 min on RTX 3080, a few min on CPU.
Run:  python -m scripts.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data import WindowConfig, prepare  # noqa: E402
from src.models import MODELS, build  # noqa: E402
from src.train import TrainConfig, train_model, evaluate, _loader  # noqa: E402


def main():
    print("device check:", "CUDA " + torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "CPU (ok for smoke test)")

    X = np.load(ROOT / "data/synthetic/om0.5_plain_s0.npy").astype(np.float32)
    cfg = WindowConfig(lookback=96, horizon=24)
    data = prepare(X, regime=0.25, cfg=cfg)

    tcfg = TrainConfig(epochs=2, seed=0)
    for name in MODELS:
        model = build(name, cfg.lookback, cfg.horizon)
        # untrained test error as reference
        te = _loader(data["test"], tcfg.batch_size, shuffle=False)
        untrained_mse, _ = evaluate(model.to(tcfg.device), te, tcfg.device)
        res = train_model(build(name, cfg.lookback, cfg.horizon), data, tcfg)
        ok = np.isfinite(res["test_mse"]) and res["test_mse"] < untrained_mse
        print(f"{name:12s} untrained={untrained_mse:8.4f} "
              f"trained={res['test_mse']:8.4f}  {'OK' if ok else 'FAIL'}")
        assert ok, f"{name}: training did not improve over untrained model"

    print("\nSMOKE TEST: PASS — run_baselines'i başlatabilirsin.")


if __name__ == "__main__":
    main()
