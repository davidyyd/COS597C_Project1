# Low-rank self-gate at rank 48
Parent: r08b-lowrank-selfgate
Type: exploit

Is the per-dimension self-gate capacity-limited? Same mechanism, rank 32 -> 48 per head (attention params
still under cap). Confirm: < r08b by > 0.0005 means rank matters; else rank 32 suffices.
