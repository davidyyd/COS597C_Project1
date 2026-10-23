# Output temperature after the soft-cap
Parent: r28a-best-outtemp
Type: exploit

The soft-cap is essential (removing it: +0.006, r33b). In r28a the per-token temperature scales logits before
the cap, so it cannot raise the maximum logit beyond 15. Apply it after the cap instead:
logits = 15 (1 + tanh(w . x)) tanh(l / 15): confident tokens can exceed the cap (up to 30), uncertain ones
flatten. Confirm: < r28a (0.94335).
