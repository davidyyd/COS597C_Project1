# Hard document masking (test of a known method as a possible base)
Parent: baseline. Type: explore.

Hypothesis: rows are best-fit packs of several documents; attending across document boundaries adds
noise. My soft variant (r01b) was too slow (learnable score_mod bias); this tests the plain hard mask to
see how large the effect is in this setup and whether it is worth building on (it also makes attention
sparser, which may buy time for other changes).

Change: flex_attention block mask = causal & same-document (& window), doc = cumsum(BOS), built per batch.
Confirm: < baseline. Closest work: intra-document masking (Llama 3, modded-nanogpt). Not novel by itself.
