# Query temperature with a log-position term
Parent: r05b-best-x0gate
Type: exploit

The per-token query temperature 1 + tanh(W x) cannot see how many keys compete (RoPE is relative; x has no
absolute position). Softmax entropy grows with log(n), so add a per-head term on log(1+t)/log(T):
q *= 1 + tanh(W x + b_h * log(1+t)/log(T)), b init 0.
Confirm: < r05b by > 0.0005.
Closest work: scalable softmax / log-n scaling (fixed s log n); here learned per head and combined with the
content-dependent temperature, bounded.
