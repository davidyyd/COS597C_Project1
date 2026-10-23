# Output temperature with a token-specific term from a normalized sparse table
Parent: r43a-base-valuepath
Type: explore

Per-token output temperature (content-based) helped; how confident the next-token distribution should be also
depends strongly on the current token itself (after "the" vs after "." vs mid-word). Add a token-specific
term from a small normalized table: logits *= 1 + tanh(w . norm(x_L) + c . norm(E[idx_t])) (E 8192x64, w, c init 0).
Confirm: < r43a (0.94690) by > 0.001 (content-only out-temp gave ~-0.002 on older models).
