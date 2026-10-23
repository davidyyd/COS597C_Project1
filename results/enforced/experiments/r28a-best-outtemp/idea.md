# Best model + per-token output temperature
Parent: r26a-vegate-full
Type: exploit

The content-based output temperature (logits *= 1 + tanh(w . norm(x_L))) helped baseline (-0.0025) but hurt
on the content-q-temperature model (r05a). r26a has no content q-temperature (doc temperature only); retest.
Confirm: < r26a (0.94391) by > 0.0005.
