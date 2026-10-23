# r06a with the head gate applied in bf16
Parent: r06a-stalegates-headgate. Type: exploit.

r06a (stale gates + temperature + head output gate) reached 0.9482 but 265 s (cap 261.7). The flex
output is fp32 here (fp32 RoPE tables make q,k fp32), so the gate multiply ran on fp32 tensors. c_proj
casts its input to bf16 under autocast anyway, so gating after a bf16 cast is numerically the same
model; this is an implementation fix of my own addition, to fit the cap.
Confirm: ≈ 0.9482 at < 261.7 s.
