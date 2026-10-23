# Conv layers 0-1 + stale gates + temperature + key/value smear + head gate
Parent: r10c-r09c-vsmear (adds r10b's conv in layer 1). Type: exploit.

r10c (value smear) reached 0.9434 but at 269 s; conv in layer 1 saved 13 s for +0.0008 in r10b.
Trading the layer-1 attention for value smear should give the best model within the cap.
Confirm: < 0.9445 at < 261.7 s.
