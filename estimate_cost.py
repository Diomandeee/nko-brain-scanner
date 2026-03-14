#!/usr/bin/env python3
"""Quick cost estimation for Vast.ai experiments."""

# Vast.ai pricing (approximate, March 2026)
HOURLY_RATE = 1.80  # 2x A100 80GB SXM, $/hr

# Experiment 1: Activation profiling
# 100 corpus pairs x 2 scripts x ~3s per forward pass = ~600s = ~10min
# Add overhead (model loading, checkpointing): ~30min total
EXP1_HOURS = 0.5

# Experiment 2: Coarse heatmap sweep
# Coarse configs: ~200 configs per script (every 4th layer)
# Each config: duplicate layers + 32 probes x ~5s = ~160s per config
# 200 configs x 2 scripts x 160s = 64,000s = ~18 hours
# But generation is the bottleneck, not duplication, so roughly:
# 200 configs x 2 scripts x ~90s = 36,000s = ~10 hours
EXP2_COARSE_HOURS = 10

# Experiment 2: Full sweep (only if coarse shows promise)
# ~3,000 configs per script x 2 scripts x ~90s = 540,000s = ~150 hours
# This is expensive — $270. Only do if coarse results justify it.
EXP2_FULL_HOURS = 150

print("=== NKO Brain Scanner — Cost Estimate ===")
print()
print(f"Instance: 2x A100 80GB SXM @ ${HOURLY_RATE:.2f}/hr")
print()
print(f"Experiment 1 (Activation Profiling):")
print(f"  Duration: ~{EXP1_HOURS:.1f} hours")
print(f"  Cost:     ~${EXP1_HOURS * HOURLY_RATE:.2f}")
print()
print(f"Experiment 2 (Coarse Heatmap Sweep):")
print(f"  Duration: ~{EXP2_COARSE_HOURS} hours")
print(f"  Cost:     ~${EXP2_COARSE_HOURS * HOURLY_RATE:.2f}")
print()
print(f"Experiment 2 (Full Heatmap Sweep) [OPTIONAL]:")
print(f"  Duration: ~{EXP2_FULL_HOURS} hours")
print(f"  Cost:     ~${EXP2_FULL_HOURS * HOURLY_RATE:.2f}")
print()
print(f"Recommended path (coarse first):")
print(f"  Total:    ~{EXP1_HOURS + EXP2_COARSE_HOURS:.0f} hours")
print(f"  Cost:     ~${(EXP1_HOURS + EXP2_COARSE_HOURS) * HOURLY_RATE:.2f}")
print()
print("Strategy: Run Experiment 1 first (~$1). If activation profiles")
print("show meaningful EN/NKO differences, proceed to coarse sweep (~$18).")
print("Only run full sweep if coarse heatmap reveals a compelling signal.")
