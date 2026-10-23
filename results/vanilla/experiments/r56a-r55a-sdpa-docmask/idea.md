# r55a with the full-layer document mask applied through SDPA (boolean attn_mask) instead of flex
Parent: r55a-r52a-nomlpgate. Type: exploit (implementation / speed).
Same model; the doc mask in the two global layers is passed to SDPA as a (B,1,T,T) boolean mask instead of
a flex score_mod. If faster, the freed time can bring back the MLP gate (worth ~0.001).
Confirm: same val as r55a, faster.
