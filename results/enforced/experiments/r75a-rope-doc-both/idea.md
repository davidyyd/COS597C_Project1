# Combine: RoPE base ∝ window + doc position in output temperature + doc position in x0 gate
Parent: r74b-rope-scale-full
Type: exploit

Hypothesis: the round-74 doc-position additions (out temp -0.0002, x0 gate -0.0001 on r73d) are small but free,
and enter different places (output confidence vs re-grounding); combined on the new best they should add up.
Change: logits *= 1+tanh(w·x + c·f_doc); x0 gate g_i = 1+tanh(w_i·norm(x) + c_i·f_doc).
Confirm: below r74b (0.93121) by > 0.0003.
