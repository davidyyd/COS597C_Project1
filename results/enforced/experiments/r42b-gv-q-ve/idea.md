# Queries read the normalized value embedding (value-path world)
Parent: r39d-base-gated-values
Type: explore (re-test in the fixed-VE model)

Queries reading token identity helped the baseline (r17b). In VE layers: q <- norm(q) + tanh(W x) norm(VE[t])
before RoPE and QK-norm (signed gate, 0 at init). Confirm: < r39d (0.94631).
