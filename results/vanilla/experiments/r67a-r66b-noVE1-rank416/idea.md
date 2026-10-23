# r66b trading layer 1's value-embedding table (4.2M params) for bigram-prior rank 416
Parent: r66b-r65a-rank160 (0.927071). Type: exploit.
The bigram prior's returns per parameter were steep (rank 64 -> 128: -0.001; 128 -> 160: -0.0007) and the
parameter cap binds. Layer 1 is a local (128-window) layer whose values already see contextual embeddings.
Params unchanged (52.71M). Confirm: < 0.927071 within the caps.
