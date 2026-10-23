# r19b + flat bf16 grad all_reduce (clean speed/quality test)

Parent: r19b-d21-bigram-backout (0.77822, 3217s train). Efficiency: replace ~150 per-parameter fp32 all_reduces by one all_reduce of a flat bf16 copy of all grads (half the bytes). Checks speed and whether bf16 grad summation costs quality.
