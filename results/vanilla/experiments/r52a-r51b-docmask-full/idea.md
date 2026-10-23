# r51b + document masking in the two full-attention layers (non-learnable score_mod)
Parent: r51b-r49a-pointer. Type: exploit.
Hard document masking helped the baseline (r05b, -0.0022) but was too slow with per-batch block masks and
broke with in-graph block masks. Here only the two global layers (where cross-document attention is most
common; the others are local or windowed) use flex attention with a static causal block mask and a
score_mod that sends other-document keys to -inf (captured doc ids, no learnable tensors).
Confirm: < 0.930749 within the cap.
