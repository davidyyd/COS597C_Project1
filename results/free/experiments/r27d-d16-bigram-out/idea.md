# d16: output-side bigram table added to the final hidden state

Parent: r26d-d16-mu09-lr12e3 (0.78423). Explore: a second hashed bigram table (65521 x d, zero init, embedding LR, no WD) added to the residual stream right before the final norm/lm_head, a direct (t-1, t) -> next-token pathway the deep stack does not have to carry.
