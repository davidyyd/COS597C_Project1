# Bigram queries and keys: lag-1 smear on both q and k
Parent: r25a-bf16-resid. Type: explore.

Hypothesis: copying/induction needs a query "what came after my context?" to find a key "my predecessor
was X". Smearing keys alone (r02d) helped; smearing queries as well lets a head match on token pairs
(the query encodes (t-1, t), keys encode (s-1, s)), i.e. bigram-conditioned retrieval in a single layer.
Change: q_t <- q_t + b ⊙ q_{t-1}, k_t <- k_t + a ⊙ k_{t-1} (per channel, init 0).
Confirm: < 0.95726 (r25a). Closest work: smeared keys (Olsson et al. 2022), RWKV token shift.
Difference: smearing the queries too, for pair-conditioned matching.
