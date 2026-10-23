# Ablation: best without the low-rank head self-gate (odd layers)
Parent: r64a-ingate-instead
Type: exploit (ablation / new base)

Purpose: measure what the rank-32 per-head self-gate in odd layers is worth on the best model (it costs ~1.6 ms of
the step budget). If small, this becomes a base with ~2.5 ms headroom for ideas that currently fail the time cap.
Change: lowrank = False (the scalar head self-gate a_h . y_h stays).
