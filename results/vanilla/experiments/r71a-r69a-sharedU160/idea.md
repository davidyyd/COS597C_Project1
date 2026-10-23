# r69a with one token table shared by both lags (separate output factors), rank 160 each
Parent: r69a-r68a-split192-96. Type: exploit.
logits += U[x_t] V^T + U[x_(t-1)] V2^T: the token table is shared across lags, so the same parameters give
rank 160 per lag (vs 192 + 96). Confirm: < 0.926560 within the caps.
