# Selective contextual embedding: token-dependent lag gates in the contextual x0
Parent: r11b-context-x0. Type: explore.

Hypothesis: how much a token should absorb its predecessors' embeddings depends on the token (a word-piece
continuation should absorb the previous pieces, a sentence start should not). Making the lag weights of
the contextual x0 conv token-dependent (a gate per lag read from the token's own embedding) turns the
fixed conv into a cheap selective one, computed once at the input.
Change: x0 = norm(sum_j g_j(e_t) w_j ⊙ e_{t-j}), g_j = 2 sigmoid(a_j · norm(e_t[:32])), a init 0.
Confirm: < 0.9533 (r11b). Refute: no gain.
Closest work: selective SSMs (Mamba: input-dependent dynamics), short convs in Hyena/Canon. Difference:
input-dependent lag gates on an embedding-level conv.
