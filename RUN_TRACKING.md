# Run Tracking Sheet

Last updated: 2026-02-19

## Scope

This sheet tracks the main experiment lines:

1. `edu-d26-top50` (extra quality filter on full filtered dataset)
2. `edu-d26-top20` (extra quality filter on full filtered dataset)
3. `exp-top16-d26-8gpu` (full filtered dataset run, split across multiple W&B runs)
4. `exp-rand16-d26-8gpu` (random/staging data arm on 8x H100)

## Run Registry

| Run label | W&B run name | W&B run ID(s) | Host | GPU setup | Data variant | Start (UTC) | W&B URL |
|---|---|---|---|---|---|---|---|
| top50 | `edu-d26-top50` | `zoebqeam` | `192.222.50.180` | 1x NVIDIA GH200 480GB | Full filtered + extra top-50% quality layer | 2026-02-17 20:46:50.105547 | https://wandb.ai/fabdata/nanochat/runs/zoebqeam |
| top20 | `edu-d26-top20` | `pb0ezqrx` | `192.222.58.132` | 1x NVIDIA GH200 480GB | Full filtered + extra top-20% quality layer | 2026-02-17 16:09:01.016072 | https://wandb.ai/fabdata/nanochat/runs/pb0ezqrx |
| full-filtered (split) | `exp-top16-d26-8gpu` | `dbwtodle` + `n9hithr0` + `bk8zeag4` | `192.222.53.38` | 8x NVIDIA H100 80GB HBM3 | Full filtered (no extra top20/top50 layer) | 2026-02-18 13:20:09.311701 + 2026-02-18 14:26:27.369938 + 2026-02-19 05:39:29.079169 | https://wandb.ai/fabdata/nanochat/runs/dbwtodle + https://wandb.ai/fabdata/nanochat/runs/n9hithr0 + https://wandb.ai/fabdata/nanochat/runs/bk8zeag4 |
| random-8gpu (split) | `exp-rand16-d26-8gpu` | `aw30lam9` + `4ul335c6` | `192.222.53.38` | 8x NVIDIA H100 80GB HBM3 | Random/staging subset | 2026-02-18 20:00:05.557483 + 2026-02-18 21:05:28.394977 | https://wandb.ai/fabdata/nanochat/runs/aw30lam9 + https://wandb.ai/fabdata/nanochat/runs/4ul335c6 |

## Comparability Notes

- Same model depth across these runs: `d26`.
- `top50` and `top20` use single-GPU GH200, auto batch size `1,048,576` tokens, and computed horizon `9,196` iterations (`9,642,704,896` tokens).
- `exp-top16-d26-8gpu` uses fixed `--total-batch-size 524,288` and `--num-iterations 94,540` (`49,566,187,520` tokens planned), so it is a much longer-horizon run.
- The 8x run is split:
  - Phase A: `dbwtodle` (no CDPK integrated)
  - Phase B: `n9hithr0` (`--resume-from-step 2000`, CDPK integrated every 1000 steps)
  - Phase C: `bk8zeag4` (`--resume-from-step 12000`, CDPK integrated every 1000 steps)
- `exp-rand16-d26-8gpu` had an early failed startup run (`45vxrhi8`, CUDA OOM), then successful split phases (`aw30lam9` and `4ul335c6`).

## Current Headline Snapshot

| Run label | Latest training step seen | Best validation bpb seen | Best CORE seen | Best CDPK seen | Notes |
|---|---:|---:|---:|---:|---|
| top50 (`zoebqeam`) | 8593 | 0.633631 (step 8500) | 0.2154 (step 8000) | 0.2892 (260/899, step 2000) | CDPK/CORE every 1000 steps |
| top20 (`pb0ezqrx`) | 9195 | 0.581783 (step 9196) | 0.2400 (step 9196) | 0.2881 (259/899, step 5000) | CDPK/CORE every 1000 steps |
| full-filtered split (`dbwtodle` + `n9hithr0` + `bk8zeag4`) | 12359 (latest phase) | 0.626348 (step 12250) | 0.2221 (step 10000) | 0.2948 (265/899, step 11000) | Phase A reached step 2347; phase B resumed at 2000; phase C resumed at 12000 |
| random-8gpu split (`aw30lam9` + `4ul335c6`) | 19391 | 0.788619 (step 19250) | 0.2636 (step 19000) | 0.2948 (265/899, step 13000) | Phase A reached step 1740, then resumed from checkpoint 1000 in phase B |

## Milestone Table (fill as runs progress)

Use this for side-by-side comparisons at common checkpoints.

| Run label | Step | Validation bpb | CORE | CDPK accuracy | Comment |
|---|---:|---:|---:|---:|---|
| top50 | 250 | 0.946652 | n/a | n/a | |
| top50 | 500 | 0.861244 | n/a | n/a | |
| top50 | 750 | 0.824917 | n/a | n/a | |
| top50 | 1000 | 0.800964 | 0.1280 | 0.2803 | |
| top50 | 1250 | 0.785501 | n/a | n/a | |
| top50 | 1500 | 0.768520 | n/a | n/a | |
| top50 | 1750 | 0.755276 | n/a | n/a | |
| top50 | 2000 | 0.747830 | 0.1577 | 0.2892 | |
| top50 | 2250 | 0.742024 | n/a | n/a | |
| top50 | 2500 | 0.736204 | n/a | n/a | |
| top50 | 2750 | 0.725109 | n/a | n/a | |
| top50 | 3000 | 0.723018 | 0.1854 | 0.2770 | |
| top50 | 3250 | 0.714976 | n/a | n/a | |
| top50 | 3500 | 0.713207 | n/a | n/a | |
| top50 | 3750 | 0.704395 | n/a | n/a | |
| top50 | 4000 | 0.704092 | 0.1932 | 0.2814 | |
| top50 | 4250 | 0.697638 | n/a | n/a | |
| top50 | 4500 | 0.697746 | n/a | n/a | |
| top50 | 4750 | 0.689900 | n/a | n/a | |
| top50 | 5000 | 0.686484 | 0.2059 | 0.2747 | |
| top50 | 5250 | 0.681798 | n/a | n/a | |
| top50 | 5500 | 0.675846 | n/a | n/a | |
| top50 | 5750 | 0.671506 | n/a | n/a | |
| top50 | 6000 | 0.668497 | 0.2036 | 0.2848 | |
| top50 | 6250 | 0.664685 | n/a | n/a | |
| top50 | 6500 | 0.661504 | n/a | n/a | |
| top50 | 6750 | 0.656067 | n/a | n/a | |
| top50 | 7000 | 0.652993 | 0.2067 | 0.2859 | |
| top50 | 7250 | 0.649631 | n/a | n/a | |
| top50 | 7500 | 0.645567 | n/a | n/a | |
| top50 | 7750 | 0.642125 | n/a | n/a | |
| top50 | 8000 | 0.639461 | 0.2154 | 0.2803 | |
| top50 | 8250 | 0.636351 | n/a | n/a | |
| top50 | 8500 | 0.633631 | n/a | n/a | |
| top20 | 250 | 0.915232 | n/a | n/a | |
| top20 | 500 | 0.811242 | n/a | n/a | |
| top20 | 750 | 0.776285 | n/a | n/a | |
| top20 | 1000 | 0.753023 | 0.1381 | 0.2681 | |
| top20 | 1250 | 0.741315 | n/a | n/a | |
| top20 | 1500 | 0.723184 | n/a | n/a | |
| top20 | 1750 | 0.712343 | n/a | n/a | |
| top20 | 2000 | 0.702149 | 0.1481 | 0.2747 | |
| top20 | 2250 | 0.698597 | n/a | n/a | |
| top20 | 2500 | 0.689661 | n/a | n/a | |
| top20 | 2750 | 0.682920 | n/a | n/a | |
| top20 | 3000 | 0.676080 | 0.1806 | 0.2836 | |
| top20 | 3250 | 0.671525 | n/a | n/a | |
| top20 | 3500 | 0.664314 | n/a | n/a | |
| top20 | 3750 | 0.657186 | n/a | n/a | |
| top20 | 4000 | 0.652731 | 0.1977 | 0.2714 | |
| top20 | 4250 | 0.647650 | n/a | n/a | |
| top20 | 4500 | 0.645243 | n/a | n/a | |
| top20 | 4750 | 0.638002 | n/a | n/a | |
| top20 | 5000 | 0.631056 | 0.2012 | 0.2881 | |
| top20 | 5250 | 0.622973 | n/a | n/a | |
| top20 | 5500 | 0.616559 | n/a | n/a | |
| top20 | 5750 | 0.607126 | n/a | n/a | |
| top20 | 6000 | 0.599841 | 0.2113 | 0.2770 | |
| top20 | 6250 | 0.591462 | n/a | n/a | |
| top20 | 6500 | 0.597181 | n/a | n/a | |
| top20 | 6750 | 0.605425 | n/a | n/a | |
| top20 | 7000 | 0.605500 | 0.2284 | 0.2670 | |
| top20 | 7250 | 0.604620 | n/a | n/a | |
| top20 | 7500 | 0.601490 | n/a | n/a | |
| top20 | 7750 | 0.598151 | n/a | n/a | |
| top20 | 8000 | 0.595498 | 0.2294 | 0.2759 | |
| top20 | 8250 | 0.592080 | n/a | n/a | |
| top20 | 8500 | 0.588619 | n/a | n/a | |
| top20 | 8750 | 0.585529 | n/a | n/a | |
| top20 | 9000 | 0.582885 | 0.2395 | 0.2714 | |
| top20 | 9196 | 0.581783 | 0.2400 | 0.2759 | |
| full-filtered (split) | 250 | 0.898655 | n/a | n/a | phase A |
| full-filtered (split) | 500 | 0.782442 | n/a | n/a | phase A |
| full-filtered (split) | 750 | 0.745609 | n/a | n/a | phase A |
| full-filtered (split) | 1000 | 0.723263 | 0.1264 | n/a | phase A |
| full-filtered (split) | 1250 | 0.709721 | n/a | n/a | phase A |
| full-filtered (split) | 1500 | 0.700635 | n/a | n/a | phase A |
| full-filtered (split) | 1750 | 0.691277 | n/a | n/a | phase A |
| full-filtered (split) | 2000 | 0.684965 | 0.1641 | 0.2781 | phase A |
| full-filtered (split) | 2250 | 0.680117 | n/a | n/a | phase A |
| full-filtered (split) | 2500 | 0.676328 | n/a | n/a | phase B |
| full-filtered (split) | 2750 | 0.673561 | n/a | n/a | phase B |
| full-filtered (split) | 3000 | 0.670461 | 0.1853 | 0.2914 | phase B |
| full-filtered (split) | 3250 | 0.671822 | n/a | n/a | phase B |
| full-filtered (split) | 3500 | 0.670284 | n/a | n/a | phase B |
| full-filtered (split) | 3750 | 0.669489 | n/a | n/a | phase B |
| full-filtered (split) | 4000 | 0.676572 | 0.1860 | 0.2803 | phase B |
| full-filtered (split) | 4250 | 0.680611 | n/a | n/a | phase B |
| full-filtered (split) | 4500 | 0.686164 | n/a | n/a | phase B |
| full-filtered (split) | 4750 | 0.684007 | n/a | n/a | phase B |
| full-filtered (split) | 5000 | 0.688860 | 0.1835 | 0.2692 | phase B |
| full-filtered (split) | 5250 | 0.687016 | n/a | n/a | phase B |
| full-filtered (split) | 5500 | 0.696292 | n/a | n/a | phase B |
| full-filtered (split) | 5750 | 0.695808 | n/a | n/a | phase B |
| full-filtered (split) | 6000 | 0.689664 | 0.1653 | 0.2736 | phase B |
| full-filtered (split) | 6250 | 0.681745 | n/a | n/a | phase B |
| full-filtered (split) | 6500 | 0.681465 | n/a | n/a | phase B |
| full-filtered (split) | 6750 | 0.670046 | n/a | n/a | phase B |
| full-filtered (split) | 7000 | 0.664711 | 0.1839 | 0.2881 | phase B |
| full-filtered (split) | 7250 | 0.659319 | n/a | n/a | phase B |
| full-filtered (split) | 7500 | 0.655901 | n/a | n/a | phase B |
| full-filtered (split) | 7750 | 0.654572 | n/a | n/a | phase B |
| full-filtered (split) | 8000 | 0.651849 | 0.2126 | 0.2747 | phase B |
| full-filtered (split) | 8250 | 0.647972 | n/a | n/a | phase B |
| full-filtered (split) | 8500 | 0.646015 | n/a | n/a | phase B |
| full-filtered (split) | 8750 | 0.644328 | n/a | n/a | phase B |
| full-filtered (split) | 9000 | 0.642506 | 0.2178 | 0.2770 | phase B |
| full-filtered (split) | 9250 | 0.639268 | n/a | n/a | phase B |
| full-filtered (split) | 9500 | 0.637479 | n/a | n/a | phase B |
| full-filtered (split) | 9750 | 0.636238 | n/a | n/a | phase B |
| full-filtered (split) | 10000 | 0.633873 | 0.2221 | 0.2792 | phase B |
| full-filtered (split) | 10250 | 0.633310 | n/a | n/a | phase B |
| full-filtered (split) | 10500 | 0.632106 | n/a | n/a | phase B |
| full-filtered (split) | 10750 | 0.631397 | n/a | n/a | phase B |
| full-filtered (split) | 11000 | 0.629150 | 0.2150 | 0.2948 | phase B |
| full-filtered (split) | 11250 | 0.628348 | n/a | n/a | phase B |
| full-filtered (split) | 11500 | 0.628390 | n/a | n/a | phase B |
| full-filtered (split) | 11750 | 0.627681 | n/a | n/a | phase B |
| full-filtered (split) | 12000 | 0.626479 | 0.2180 | 0.2859 | phase B |
| full-filtered (split) | 12250 | 0.626348 | n/a | n/a | phase C |
| random-8gpu (split) | 250 | 1.151184 | n/a | n/a | phase A |
| random-8gpu (split) | 500 | 1.035174 | n/a | n/a | phase A |
| random-8gpu (split) | 750 | 0.991594 | n/a | n/a | phase A |
| random-8gpu (split) | 1000 | 0.968255 | 0.1160 | 0.2692 | phase A |
| random-8gpu (split) | 1250 | 0.959076 | n/a | n/a | phase A |
| random-8gpu (split) | 1500 | 0.941521 | n/a | n/a | phase A |
| random-8gpu (split) | 1750 | 0.930998 | n/a | n/a | phase B |
| random-8gpu (split) | 2000 | 0.922632 | 0.1552 | 0.2670 | phase B |
| random-8gpu (split) | 2250 | 0.915536 | n/a | n/a | phase B |
| random-8gpu (split) | 2500 | 0.908176 | n/a | n/a | phase B |
| random-8gpu (split) | 2750 | 0.903801 | n/a | n/a | phase B |
| random-8gpu (split) | 3000 | 0.898076 | 0.1903 | 0.2770 | phase B |
| random-8gpu (split) | 3250 | 0.895019 | n/a | n/a | phase B |
| random-8gpu (split) | 3500 | 0.891521 | n/a | n/a | phase B |
| random-8gpu (split) | 3750 | 0.886668 | n/a | n/a | phase B |
| random-8gpu (split) | 4000 | 0.885078 | 0.2000 | 0.2703 | phase B |
| random-8gpu (split) | 4250 | 0.882274 | n/a | n/a | phase B |
| random-8gpu (split) | 4500 | 0.880342 | n/a | n/a | phase B |
| random-8gpu (split) | 4750 | 0.878368 | n/a | n/a | phase B |
| random-8gpu (split) | 5000 | 0.876222 | 0.1838 | 0.2447 | phase B |
| random-8gpu (split) | 5250 | 0.873927 | n/a | n/a | phase B |
| random-8gpu (split) | 5500 | 0.872271 | n/a | n/a | phase B |
| random-8gpu (split) | 5750 | 0.870300 | n/a | n/a | phase B |
| random-8gpu (split) | 6000 | 0.870165 | 0.2013 | 0.2803 | phase B |
| random-8gpu (split) | 6250 | 0.867909 | n/a | n/a | phase B |
| random-8gpu (split) | 6500 | 0.867256 | n/a | n/a | phase B |
| random-8gpu (split) | 6750 | 0.868148 | n/a | n/a | phase B |
| random-8gpu (split) | 7000 | 0.866301 | 0.2002 | 0.2681 | phase B |
| random-8gpu (split) | 7250 | 0.865605 | n/a | n/a | phase B |
| random-8gpu (split) | 7500 | 0.865505 | n/a | n/a | phase B |
| random-8gpu (split) | 7750 | 0.864945 | n/a | n/a | phase B |
| random-8gpu (split) | 8000 | 0.863745 | 0.2038 | 0.2570 | phase B |
| random-8gpu (split) | 8250 | 0.863150 | n/a | n/a | phase B |
| random-8gpu (split) | 8500 | 0.862476 | n/a | n/a | phase B |
| random-8gpu (split) | 8750 | 0.861714 | n/a | n/a | phase B |
| random-8gpu (split) | 9000 | 0.862859 | 0.2080 | 0.2781 | phase B |
| random-8gpu (split) | 9250 | 0.860735 | n/a | n/a | phase B |
| random-8gpu (split) | 9500 | 0.860023 | n/a | n/a | phase B |
| random-8gpu (split) | 9750 | 0.860498 | n/a | n/a | phase B |
| random-8gpu (split) | 10000 | 0.858284 | 0.2076 | 0.2792 | phase B |
| random-8gpu (split) | 10250 | 0.857160 | n/a | n/a | phase B |
| random-8gpu (split) | 10500 | 0.853628 | n/a | n/a | phase B |
| random-8gpu (split) | 10750 | 0.855319 | n/a | n/a | phase B |
| random-8gpu (split) | 11000 | 0.851616 | 0.2160 | 0.2825 | phase B |
| random-8gpu (split) | 11250 | 0.851302 | n/a | n/a | phase B |
| random-8gpu (split) | 11500 | 0.850004 | n/a | n/a | phase B |
| random-8gpu (split) | 11750 | 0.847945 | n/a | n/a | phase B |
| random-8gpu (split) | 12000 | 0.844113 | 0.2163 | 0.2770 | phase B |
| random-8gpu (split) | 12250 | 0.839243 | n/a | n/a | phase B |
| random-8gpu (split) | 12500 | 0.838259 | n/a | n/a | phase B |
| random-8gpu (split) | 12750 | 0.838303 | n/a | n/a | phase B |
| random-8gpu (split) | 13000 | 0.838301 | 0.2206 | 0.2948 | phase B |
| random-8gpu (split) | 13250 | 0.834635 | n/a | n/a | phase B |
| random-8gpu (split) | 13500 | 0.834087 | n/a | n/a | phase B |
| random-8gpu (split) | 13750 | 0.833664 | n/a | n/a | phase B |
| random-8gpu (split) | 14000 | 0.828823 | 0.2184 | 0.2781 | phase B |
| random-8gpu (split) | 14250 | 0.828875 | n/a | n/a | phase B |
| random-8gpu (split) | 14500 | 0.828261 | n/a | n/a | phase B |
| random-8gpu (split) | 14750 | 0.823749 | n/a | n/a | phase B |
| random-8gpu (split) | 15000 | 0.824079 | 0.2165 | 0.2714 | phase B |
| random-8gpu (split) | 15250 | 0.822903 | n/a | n/a | phase B |
| random-8gpu (split) | 15500 | 0.818657 | n/a | n/a | phase B |
| random-8gpu (split) | 15750 | 0.817199 | n/a | n/a | phase B |
| random-8gpu (split) | 16000 | 0.814735 | 0.2305 | 0.2714 | phase B |
| random-8gpu (split) | 16250 | 0.813503 | n/a | n/a | phase B |
| random-8gpu (split) | 16500 | 0.812270 | n/a | n/a | phase B |
| random-8gpu (split) | 16750 | 0.810295 | n/a | n/a | phase B |
| random-8gpu (split) | 17000 | 0.806373 | 0.2566 | 0.2659 | phase B |
| random-8gpu (split) | 17250 | 0.804924 | n/a | n/a | phase B |
| random-8gpu (split) | 17500 | 0.803950 | n/a | n/a | phase B |
| random-8gpu (split) | 17750 | 0.802135 | n/a | n/a | phase B |
| random-8gpu (split) | 18000 | 0.798356 | 0.2570 | 0.2692 | phase B |
| random-8gpu (split) | 18250 | 0.796960 | n/a | n/a | phase B |
| random-8gpu (split) | 18500 | 0.794067 | n/a | n/a | phase B |
| random-8gpu (split) | 18750 | 0.790843 | n/a | n/a | phase B |
| random-8gpu (split) | 19000 | 0.789740 | 0.2636 | 0.2636 | phase B |
| random-8gpu (split) | 19250 | 0.788619 | n/a | n/a | phase B |

## Update Procedure

1. Run `python update_run_tracking.py` from this repository root.
2. Review the diff in `RUN_TRACKING.md`.
3. Commit when the refresh looks correct.

