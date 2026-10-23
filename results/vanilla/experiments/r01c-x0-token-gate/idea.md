# Token-dependent gate on the x0 re-injection
Parent: baseline. Type: explore.

Hypothesis: the baseline re-adds the normalized input embedding before every block with one scalar per
layer. How much a token needs its own identity re-injected should depend on context (e.g. tokens
already "resolved" vs. ambiguous ones), so a per-token gate read from the residual stream should use x0
better.

Change: x <- r_i x + l_i * g_i(x) * x0, with g_i(x) = 2 sigmoid(w_i · norm(x)[:32]) (w_i init 0, so
g=1 at start, identical to baseline).

Confirm: val_bpb below baseline. Refute: no change (a scalar per layer is enough).

Closest work: x0/embedding skip (modded-nanogpt), value-embedding gates (same 32-channel gate trick).
Difference: gating the embedding re-injection per token.
