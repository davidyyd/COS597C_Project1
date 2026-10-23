# Multiplicative key salience after QK-norm
Parent: r25a-bf16-resid. Type: explore.
r81d's additive per-key bias (through a q/k channel) was negative and slow. Multiplicative version:
k_s *= 2 sigmoid(w_h . x_s[:32]) (zero-init gives 1). This gives each key its own temperature: salient
keys get sharper, more selective matches and dull keys get flatter ones. Confirm: < 0.95726.
Closest: QK-norm with learned gains (static per channel); ours is per token and per head.
