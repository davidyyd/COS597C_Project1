# Low-rank elementwise gate on the MLP output
Parent: r25a-bf16-resid. Type: explore.
The elementwise attention-output gate was as strong as the head gate (-0.0053). The MLP output has no
gate; a per-channel low-rank gate (32-channel read) lets each token scale individual MLP output features.
Confirm: < 0.95726. Compare: input gate -0.0022, grouped hidden gate -0.0024, per-unit hidden gate -0.0034.
