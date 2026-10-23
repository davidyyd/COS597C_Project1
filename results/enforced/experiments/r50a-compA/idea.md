# Composition A: value-path baseline + deep embed + temperatures + multi-scale/x0 gate + one read + bigram MLP bias
Parent: r43a-base-valuepath
Type: exploit (composition from the re-measured per-ms values)

Re-measured on the value-path baseline: deep embed -0.0028 (+1.6 ms), doc+out temperatures -0.0027, multi-scale
+ x0 gate -0.003, MLP-reads-attention -0.0011 (+0.7 ms), bigram token MLP bias (layers 0-3) -0.0021 (+2.6 ms).
Combine them without the head self-gate and the full read chain (~5 ms). Confirm: < r41b (~0.9369).
