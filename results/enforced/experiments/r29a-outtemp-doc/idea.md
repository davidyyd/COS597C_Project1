# Output temperature from content and document position
Parent: r28a-best-outtemp
Type: exploit

Both the content output temperature (r28a, -0.0006) and the doc-position output temperature (r28b, -0.0004)
helped r26a. Combine in one gate: logits *= 1 + tanh(w . norm(x_L) + b f_doc). Confirm: < r28a (0.94335).
