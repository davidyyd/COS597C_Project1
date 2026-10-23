# Best model without the undiluted-read chain (ablation)
Parent: r41b-best-tokgated-reads
Type: exploit (ablation / time)

The read chain was a large gain before the value-path fix (and costs ~3 ms). Is it still needed? Measures how
much of its effect was compensating the value-embedding scale problem. Expect +0.002..0.004 if still useful.
