# r18b with a width-8 contextual x0 (lags 0-7)
Parent: r18b-r17a-bf16. Type: exploit.

The contextual x0 conv runs once at the input (cheap). Tests whether a wider local window in the
re-injected embedding (8 tokens: about a word or two of BPE pieces) helps beyond 4.
Confirm: < 0.9369 within the cap.
