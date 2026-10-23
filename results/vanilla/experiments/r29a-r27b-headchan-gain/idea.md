# r27b with the per-head temperature replaced by a per-(head, channel) query gain
Parent: r27b-r27a-readout. Type: exploit.

r28a: a per-channel query gain helped the control (-0.0027) but hurt r27b when stacked on top of the
per-head temperature (r28c, +0.003). Both rescale the normalized query; one general per-(head, channel)
gain (init 1) subsumes both (temperature = its mean, band weighting = its shape) without the redundancy.
Confirm: < 0.935842 within the cap.
