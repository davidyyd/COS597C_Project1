# Learned per-head attention temperature after QK-norm
Parent: baseline. Type: explore.

Hypothesis: with RMS-normalized q and k (no gains) and the 1/sqrt(d) scale, attention logits are bounded
by sqrt(128) ~ 11.3, which caps how sharp a head can be (copy/induction heads want near one-hot
attention over 1-2k keys) and forces broad heads too. A learned per-head temperature removes the cap.

Change: q <- exp(s_h) * norm(q), s_h learned per head (init 0).

Confirm: val_bpb below baseline, learned temperatures > 1 in some heads. Refute: no gain.

Closest work: QK-norm with learnable gains (OLMo 2, nGPT s_qk scaling). Difference: a single scalar per
head as an explicit temperature, testing the sharpness-cap hypothesis.
