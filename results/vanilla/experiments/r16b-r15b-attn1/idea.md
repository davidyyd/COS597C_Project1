# r15b with attention restored in layer 1 (conv only in layer 0)
Parent: r15b-r14a-noreadout. Type: exploit.

r15b has ~10 s of headroom. In r13a/b, replacing layer 1's attention cost ~0.002 val. With contextual x0
and VE, does layer-1 attention still matter? Spending the headroom on it may be the best use.
Confirm: < 0.9397 within the cap.
