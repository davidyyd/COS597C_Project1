# Gated normalized value path on the plain baseline (ablation)
Parent: baseline
Type: explore (ablation of the value-path discovery)

The value path (VE layers: v = s(g1 norm(W_v u) + g2 norm(VE)); other layers v = s norm(W_v u); gates from the
full input) gave -0.0102 on the self-gated model r01b (r39d). Measure it on the plain baseline.
Confirm: < baseline by > 0.008.
