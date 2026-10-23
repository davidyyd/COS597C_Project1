# Best model without the input-embedding norm
Parent: r12b-best-attn-reads-mlp
Type: exploit

r12d: removing the RMSNorm on token embeddings gave -0.0017 on baseline (tokens keep a learned magnitude).
Apply to the best model (x0 = wte[idx]; the prev-token key input stays normalized). Confirm: < 0.9465.
