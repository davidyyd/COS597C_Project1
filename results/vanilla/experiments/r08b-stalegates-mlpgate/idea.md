# Stale depth gates + temperature + per-token MLP output gate
Parent: r05a-stale-gates (adds r06c-mlp-gate). Type: exploit.

Hypothesis: the MLP output gate was free (66.1 ms) and gave -0.0016 alone; it should add to r05a.
Confirm: < 0.9505 within the cap.
