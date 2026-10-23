# Conv layer 0 + stale depth gates + temperature + smeared keys
Parent: r06b-stalegates-smear (adds r08d-conv-layer0). Type: exploit.

r06b reached 0.9476 but ran 267 s (cap 261.7). r08d's conv-for-layer-0 was both better (-0.0035) and
9 s faster, so it should bring r06b under the cap and add its gain.
Confirm: < 0.9476 at < 261.7 s.
