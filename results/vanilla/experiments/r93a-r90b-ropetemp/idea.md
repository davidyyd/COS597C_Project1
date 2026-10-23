# r90b-r88c-depth2 + position-aware attention temperature folded into the query RoPE tables
Parent: r90b-r88c-depth2 (0.923036 @ 262.1). Type: exploit.
r92 showed the temperature q *= 1 + a_{l,h} log(1+t)/log(T) helps the stack (r92a 0.922549) but the extra
multiply on q cost ~3 s. RMSNorm (no gain) commutes with RoPE's rotation (norm-preserving), so normalize
first and rotate q with tables already scaled per (layer, head, position): cos_q = cos * s, sin_q = sin * s.
This is mathematically the same model as r92, with the temperature folded into tables of size (T, H, D/2)
per layer. Confirm: about r92-level quality within the time cap.
