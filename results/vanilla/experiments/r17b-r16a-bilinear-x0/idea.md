# r16a + multiplicative bigram code in the contextual x0
Parent: r16a-r15b-headveconv. Type: exploit.

Contextual x0 sums lagged embeddings, an additive code that cannot bind which token came before which
(e.g. "new york" vs. "york new" up to lag weights). An elementwise product e_t ⊙ e_{t-1} is a cheap
pair-specific code (a hash-free bigram embedding). Computed once at the input (negligible cost).
Change: x0 = norm(sum_j w_j ⊙ e_{t-j} + u ⊙ e_t ⊙ e_{t-1}), u init 0.
Confirm: < 0.9391.
