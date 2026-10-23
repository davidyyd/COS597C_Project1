# Best model without the logit soft-cap
Parent: r28a-best-outtemp
Type: exploit (simplification / time)

The baseline's 15 tanh(l/15) soft-cap costs passes over the [B,T,8192] logits. r28a already has a bounded
per-token output temperature (0, 2); test whether the cap is still needed. Confirm "not needed": within
+0.0003 of r28a (0.94335) and stable.
