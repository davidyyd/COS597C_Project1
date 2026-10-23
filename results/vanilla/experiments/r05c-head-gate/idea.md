# Per-head attention output gate from the query token
Parent: baseline. Type: explore.

Hypothesis: a head always writes a convex combination of values; when it has nothing useful to retrieve
it must dump attention on a sink. A per-token, per-head scalar gate lets it write nothing (the depth/x0
gates show per-token scalar control is valuable here).

Change: y_h <- 2 sigmoid(w_h · x[:32]) y_h (w init 0) before c_proj.
Closest work: gated attention (Qiu et al. 2025, headwise variant). Tests a known mechanism in this setup.
