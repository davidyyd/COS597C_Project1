# Word-so-far embedding: mean of the token embeddings since the last word start
Parent: r18a-bf16-rope. Type: explore.

Hypothesis: BPE splits words into pieces; the identity of the *word being spelled* is what matters, and
the model must assemble it with attention. Contextual embeddings with fixed lags helped a lot (r11b), but
words have variable length. A causal "word so far" vector, the mean of the embeddings from the last
word-start token (leading space/newline, or BOS) to t, gives a word-aligned, variable-span context feature,
computed once at the input with a cumsum and a gather.
Change: x0 = norm(e_t + u ⊙ mean_{s=start(t)..t} e_s), u per channel, init 0.
Confirm: < 0.95739 (r18a). Refute: no gain.
Closest work: hierarchical/word-level tokenization models (e.g. byte/word hybrids), pooling within words.
Difference: a cheap causal word-prefix pooling added to the token embedding.
