# r65a with the prior's input factor computed from the contextual embedding (P x0_t) instead of a table U[x_t]
Parent: r65a-r63a-bigram128-fused. Type: exploit/explore.
x0 is the contextual input embedding (lags 0-3 of the token embeddings), so (P x0_t) V^T is a low-rank
n-gram -> next-token logit prior rather than a bigram one, and P (512 x 128) replaces the 1M-parameter U
table. Confirm: < 0.927801 within the caps.
