# Cheap lineage + scalar gate + full undiluted-read chain
Parent: r21b-cheap-lineage-gate
Type: exploit

r21b (doc-temperature + MLP reads attention + scalar head self-gate) plus the remaining chain reads
(attn <- a_{l-1}, m_{l-1}; MLP <- m_{l-1}). (Without the gate the same chain measured 70.7 ms; the gated
model measures faster.) Confirm: < r21b by > 0.001.
