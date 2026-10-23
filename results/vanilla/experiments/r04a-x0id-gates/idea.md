# Depth gates read from the token embedding only (identity gates)
Parent: r03a-depthgates-temp. Type: exploit.

Hypothesis/question: are the per-token depth gates useful because of context (read from the residual x)
or because of token identity? If identity suffices, all layers' gates can be computed once from x0 in
one matmul, removing the per-layer gate read (r03a was at the time cap, 262.6 s).

Change: G = 2 sigmoid(x0[:32] W), W in R^{32 x 2L}, computed once; layer i uses G[2i:2i+2].
Confirm: ≈ r03a (0.9496) and faster. Refute: clearly worse => the gates need context.
