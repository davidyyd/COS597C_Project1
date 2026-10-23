# Self-gated undiluted read
Parent: r58b-va-gates-adam
Type: explore

MLP reads the block's attention output, with the read self-gated per channel: u + b * norm(a) * 2 sigmoid(U V norm(a)).
Compare with the plain read r68d. Confirm: < r68d.
