# r16a with layer 1 as short local attention (window 128) instead of a conv
Parent: r16a-r15b-headveconv (layer 1 from r16b). Type: exploit.

Layer-1 attention with the full 1024 window helped (r16b 0.9381) but cost 17 s; the conv was cheap but
weaker. Hypothesis: early layers need content-based but local mixing; a 128-token window gives that at a
fraction of the cost (block-sparse flex attention).
Confirm: < 0.9391 within the cap.
