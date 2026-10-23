# Low-rank bag-of-words document context in x0
Parent: r77c-docmean-x0 (on r25a-bf16-resid). Type: explore (refine of a new direction).
r77c: full-width doc mean -0.0030 vs control but +5.4 s. Rank-128 version: m_t = mean_{s in doc, s<=t} A e_s,
x = norm(e + B norm(m_t)). Tests whether the doc-context signal survives a 128-d bottleneck at lower cost.
Confirm: close to 0.954226 at less time. Closest: TopicRNN / continuous cache; linear attention with
uniform weights.
