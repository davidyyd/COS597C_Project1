# Finer RoPE in the document-isolated early layers (base 10 x window there)
Parent: r87a-rope2-doc-out
Type: exploit
Hypothesis: removing RoPE from the isolated early layers is catastrophic (r84d, +0.039), so position matters most
there, while late layers prefer coarse positions (larger bases helped). Halve the RoPE bases in layers 0-2 only
(10 x window), keep 20 x window elsewhere. Confirm: below r87a by > 0.0003.
