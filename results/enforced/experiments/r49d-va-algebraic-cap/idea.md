# Algebraic logit soft-cap (heavier tails than tanh)
Parent: r43a-base-valuepath
Type: explore

The soft-cap is essential (removing it: +0.006, r33b), so its shape may matter. tanh saturates exponentially:
logits near the cap are compressed hard. The algebraic cap l / sqrt(1 + (l/15)^2) has the same slope at 0 and
the same bound but approaches it polynomially. Confirm/refute: compare with r43a (0.94690).
Closest work: tanh logit soft-capping (Gemma 2, modded-nanogpt). Differs: algebraic sigmoid shape.
