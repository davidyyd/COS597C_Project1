# r18b without head gate + lookahead value embeddings in layer 7
Parent: r22a-r18b-noheadgate (adds r23a's mechanism). Type: exploit.

The head gate was the least time-efficient component of r18b (0.0013 for 10 s). If the lookahead values
are worth more than that, swapping them in should beat r18b within the cap.
Confirm: < 0.9369 at < 261.7 s.
