# r18b with layer 2 also local (window 128)
Parent: r18b-r17a-bf16. Type: exploit.

Tests how far "early layers are local" extends: layer 2 (a 1024-window layer) becomes a 128-window layer.
Confirm: ≤ 0.9369 (and slightly faster). Refute: worse => layer 2 needs mid-range context.
