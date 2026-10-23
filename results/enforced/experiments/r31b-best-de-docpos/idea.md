# Deep embedding also sees the document position
Parent: r28a-best-outtemp
Type: exploit

Feed the doc-position feature into the embedding MLP input (x0 = norm(e + MLP(e + f_doc w)), w init 0), so the
re-injected token features can depend non-linearly on how far into its document the token is.
Confirm: < r28a by > 0.0003.
