# Pointer head whose keys are the contextual token embeddings (not final states)
Parent: r34c-pointer-dense. Type: explore.

Hypothesis: the pointer should match "where did my current context occur before"; the final state at s
summarises the context *predicting s+1*, while the (contextual) embedding x0_s is a clean, stable code of
the local n-gram ending at s. Keys from x0 (computed once at the input) may give sharper, more reliable
matches, with the query still from the final state.
Change: kc = norm(W_k x0_s) instead of norm(W_k x_s^final).
Confirm: < 0.953150 (r34c).
