# Token table: query-temperature biases instead of MLP-gate features
Parent: r87a-rope2-doc-out
Type: exploit
The token MLP-gate features were worth ~0.0001. Per-token query-temperature biases (how sharply this token should
attend, per layer and head) may be more useful and are cheaper (32 vs 96 per token).
Confirm: below r87a by > 0.0003.
