# Per-token forget gate on the residual stream
Parent: r05b-best-x0gate
Type: exploit

The x0 gate (per-token weight on re-injected token identity) helped. Its complement: a per-token scale on
the carried residual, x <- r_i (1 + tanh(v_i . norm(x))) x + a_i g x0. Since every block reads norm(x),
shrinking x lets the next writes dominate (forget), growing it protects the past.
Confirm: < r05b (0.95219) by > 0.0005.
Closest work: highway / forget gates (LSTM), per-layer resid lambdas (static). Per-token scalar here.
