# Document-isolated early layers drop the time-shifted key read; low-rank head gate back in layer 1
Parent: r81a-dm012-lr357
Type: exploit (time trade)
Hypothesis: in the document-masked layers 0-2 the key's read of the previous position is less useful (and crosses
document starts); removing it saves time that buys back the low-rank head gate in layer 1.
Confirm: below r81a mean (0.93009).
