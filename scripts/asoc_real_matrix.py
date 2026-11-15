"""Resume-safe Phase 7 real-domain matrix; subprocess isolation per domain/regime/arch.

Pilot:
 python -m scripts.asoc_real_matrix --dry-run
 python -m scripts.asoc_real_matrix --limit 1 --gpu-index 0

Expand AFTER the pilot:
 python -m scripts.asoc_real_matrix --domains electricity,traffic --regimes 0.10,0.25,1.00 \\
    --archs dlinear,itransformer --reps 0,1,2 --pool-seeds 0,1

For three-channel iTransformer + jointly-generated VAE, select
--archs itransformer --input-mode all (not comparable with target-only).
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from scripts.asoc_real_controls import (CONTROL_CONDITIONS, CONTROL_POOL_SEED,
                                       DEFAULT_OUT, DOMAINS, KEY, ROOT, VERSION,
                                       parse_ints, read_completed, sha256)


def parse_choice_list(raw, allowed):
    xs = tuple(x.strip() for x in raw.split(","))
    if not xs or len(set(xs)) != len(xs) or not set(xs).issubset(allowed):
        raise ValueError(f"Expected unique values in {sorted(allowed)}")
    return xs


def expected_rows(*, domain, regime, arch, input_mode, reps, pool_seeds,
                  epochs, batch_size, lr, vae_epochs, latent, hidden, horizon_weight):
    for rep in reps:
        for condition in CONTROL_CONDITIONS:
            yield dict(domain=domain, regime=regime, arch=arch, input_mode=input_mode,
                       rep=rep, condition=condition, pool_seed=CONTROL_POOL_SEED,
                       epochs=epochs, batch_size=batch_size, lr=lr,
                       vae_epochs=vae_epochs, latent=latent, hidden=hidden,
                       horizon_weight=horizon_weight, experiment_version=VERSION)
        for pool_seed in pool_seeds:
            yield dict(domain=domain, regime=regime, arch=arch, input_mode=input_mode,
                       rep=rep, condition="vae_augmented", pool_seed=pool_seed,
                       epochs=epochs, batch_size=batch_size, lr=lr,
                       vae_epochs=vae_epochs, latent=latent, hidden=hidden,
                       horizon_weight=horizon_weight, experiment_version=VERSION)


def completed_rows(path: Path):
    _, frame = read_completed(path)
    fields = [k for k in KEY if k != "data_sha256"]
    return set(frame[fields].itertuples(index=False, name=None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domains", default="electricity")
    p.add_argument("--regimes", default="0.25")
    p.add_argument("--archs", default="dlinear")
    p.add_argument("--input-mode", choices=("target", "all"), default="target")
    p.add_argument("--reps", default="0")
    p.add_argument("--pool-seeds", default="0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--horizon-weight", type=float, default=0.5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Max incomplete domain/regime/arch groups")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        domains = parse_choice_list(args.domains, set(DOMAINS))
        regimes = tuple(float(s) for s in parse_choice_list(args.regimes, {"0.10", "0.25", "1.00", "0.1", "1.0", "1"}))
        if len(set(regimes)) != len(regimes):
            raise ValueError("Duplicate regimes")
        archs = parse_choice_list(args.archs, {"dlinear", "patchtst", "itransformer"})
        if args.input_mode == "all" and archs != ("itransformer",):
            raise ValueError("Joint VAE all-channel mode: --archs itransformer only")
        reps, pools = parse_ints(args.reps), parse_ints(args.pool_seeds)
        if (min(args.epochs, args.vae_epochs, args.batch_size, args.latent, args.hidden) < 1 or
                args.lr <= 0 or not 0 < args.horizon_weight < 1 or
                args.limit < 0 or args.gpu_index < 0):
            raise ValueError("Invalid budget, GPU, or limit")
    except ValueError as exc:
        p.error(str(exc))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    done = completed_rows(out)
    _, previous = read_completed(out)
    data_dir = args.data_dir if args.data_dir is not None else ROOT / "data" / "real"
    for domain in domains:
        path = data_dir / f"{domain}.npy"
        if path.is_file():
            old = previous[previous.domain == domain]
            if len(old) and not (old.data_sha256 == sha256(path)).all():
                p.error(f"Real data for {domain} have changed; audit existing results "
                        "and write to a new output CSV")
    fields = [k for k in KEY if k != "data_sha256"]
    pending = []
    for domain, regime, arch in itertools.product(domains, regimes, archs):
        expected = list(expected_rows(
            domain=domain, regime=regime, arch=arch, input_mode=args.input_mode,
            reps=reps, pool_seeds=pools, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, vae_epochs=args.vae_epochs,
            latent=args.latent, hidden=args.hidden, horizon_weight=args.horizon_weight))
        missing = [r for r in expected if tuple(r[k] for k in fields) not in done]
        if missing:
            pending.append((domain, regime, arch, len(missing)))
    print(f"PLAN: {len(domains)*len(regimes)*len(archs)} groups, "
          f"{len(pending)} incomplete; out={out}", flush=True)
    for i, (dom, regime, arch, missing) in enumerate(pending, 1):
        print(f"  {i:03d}: {dom}, regime={regime:.2f}, {arch}/{args.input_mode}: "
              f"{missing} new rows", flush=True)
    if args.dry_run or not pending:
        return
    from scripts.asoc_budget_controls import select_device
    select_device(allow_cpu=args.allow_cpu, gpu_index=args.gpu_index)
    todo = pending[:args.limit] if args.limit else pending
    for i, (domain, regime, arch, _) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {domain} {regime:.2f} {arch}", flush=True)
        cmd = [sys.executable, "-m", "scripts.asoc_real_controls",
               "--domain", domain, "--regime", str(regime), "--arch", arch,
               "--input-mode", args.input_mode, "--reps", args.reps,
               "--pool-seeds", args.pool_seeds, "--epochs", str(args.epochs),
               "--vae-epochs", str(args.vae_epochs),
               "--batch-size", str(args.batch_size), "--lr", str(args.lr),
               "--latent", str(args.latent), "--hidden", str(args.hidden),
               "--horizon-weight", str(args.horizon_weight),
               "--gpu-index", str(args.gpu_index), "--out", str(out)]
        if args.allow_cpu:
            cmd.append("--allow-cpu")
        if args.data_dir is not None:
            cmd += ["--data-dir", str(args.data_dir)]
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("Complete; rerun to skip existing conditions.", flush=True)


if __name__ == "__main__":
    main()
