# Elementwise attention-output gate reading a learned 32-dim projection of the full input
Parent: r38d-elementwise-attn-gate. Type: explore.

All my gates read the first 32 channels of the normalised input (a fixed, arbitrary slice; the network
must learn to route gate-relevant information there). A learned rank-32 read of all 512 channels
(Muon-trained 512->32 then 32->512, still cheap) may give better-informed gates. (A full-width read with an
AdamW-trained 512->2 depth gate diverged in r07b; here the read is a Muon matrix.)
Confirm: < 0.951943 (r38d).
