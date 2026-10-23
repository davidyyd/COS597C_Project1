# r77b + low-rank bag-of-words document context, MLP gate dropped
Parent: r77b-r69a-nounet-mem. Type: exploit (combine with the r77c explore win).
r77c (doc-reset running mean of wte projected into x0) gave -0.0030 on the control. Here it is low-rank
(mean of A e_s, A 512->128, then B 128->512 after a norm; linear so the mean commutes with A) to cut the
cumsum traffic 4x, added to the embedding before the ctx-x0 conv. Drop the MLP gate to pay for it.
Confirm: valid and < 0.92530.
