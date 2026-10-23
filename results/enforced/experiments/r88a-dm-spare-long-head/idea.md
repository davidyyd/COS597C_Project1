# Early document mask spares the longest-window head
Parent: r87a-rope2-doc-out
Type: exploit
Hypothesis: in the early isolated layers the short heads (128-512) gain most from isolation; the 1024-token head may
want the neighbouring document like the late layers. Head 3 of layers 0-2 keeps cross-document access.
Confirm: below r87a (0.92888) by > 0.0003.
