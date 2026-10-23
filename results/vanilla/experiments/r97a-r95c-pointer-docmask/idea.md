# r95c + pointer-sentinel copy head + document masking in the full layers
Parent: r95c-r94b-gatesall-ve3 (0.921402 @ 255.2 s). Type: exploit (revive an old in-stack synergy).
In the r51/r52 stack, the dense chunked pointer-sentinel copy head (p = a_sent p_vocab + sum_{s<t} a_ts
[x_{s+1} = y], exact, computed at the target) plus doc masking in the full layers gave -0.0015 for ~5 s; it
was dropped for time. r95c has 6.4 s of headroom. Confirm: < 0.921402 within 261.66 s.
