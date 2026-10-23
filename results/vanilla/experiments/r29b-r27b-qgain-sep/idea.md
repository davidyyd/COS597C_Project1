# r27b + per-channel query gain as a separate RMSNorm weight (as in r28a)
Parent: r27b-r27a-readout. Type: exploit (diagnostic of r28c).

r28c packed the gain into the AdamW scalar tensor as (1 + g); here it is a separate per-layer parameter
fused into the RMSNorm, exactly as in the control where it helped. Tells whether r28c's loss came from
the parametrization or from the interaction with the temperature.
