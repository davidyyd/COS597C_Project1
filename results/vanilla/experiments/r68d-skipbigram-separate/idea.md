# Skip-bigram logit prior with its own output factor (previous token -> next token)
Parent: r63d-bigram-logit-prior. Type: explore.
r64d added the previous token's factor through the bigram's output factor V (worse). Here the previous
token gets separate input and output factors: logits += U[x_t] V^T + U2[x_{t-1}] V2^T (rank 64 each,
folded into the head matmul). Confirm: < 0.954901 (r63d).
