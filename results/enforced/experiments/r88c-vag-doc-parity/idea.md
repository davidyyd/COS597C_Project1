# Soft current-vs-neighbour document bias via document-parity q/k features
Parent: r83a-vag-dm-ingate
Type: explore
Hypothesis: late layers use the current and the neighbouring document (r84c); hard masking hurts them. A learned,
soft preference is cheap if documents are labelled by parity s = +-1 (alternating per document): adding s_t p_q to
queries and s_j p_k to keys (per head, zero-init) adds s_t s_j (p_q.p_k) to scores: a learnable bonus/penalty for
the neighbouring document (and cross terms), per head and layer.
Change (layers 4-7): q_h += s_t p_q[h], k_h += s_j p_k[h] after QK-norm.
Confirm: below r83a (0.94006) by > 0.0005.
