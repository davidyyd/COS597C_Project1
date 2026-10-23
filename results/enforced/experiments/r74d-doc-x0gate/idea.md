# x0 re-injection gate also depends on document position
Parent: r73d-best-rope-per-window
Type: explore

Hypothesis: how much raw token identity to re-inject should depend on how much document context exists
(early in a document the token itself is most of the information). r21d showed a doc-position x0 scale helps the
baseline (-0.0018); here it enters the existing bounded re-grounding gate.
Change: g_i = 1 + tanh(w_i · norm(x) + c_i · f_doc), c_i per layer init 0.
Confirm: below r73d by > 0.0005.
