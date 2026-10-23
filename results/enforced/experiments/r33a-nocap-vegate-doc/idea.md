# Value-embedding gate also reads the document position (on the no-soft-cap model)
Parent: r33b-best-nocap
Type: exploit

VE gate = 2 sigmoid(W x + c f_doc) (c per head, init 0). (On r28a it measured 68.9 ms; r33b is ~0.7 ms faster.)
Confirm: < r33b by > 0.0003.
