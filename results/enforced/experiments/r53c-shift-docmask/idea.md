# Time-shifted read masked at document boundaries
Parent: r51d-c-shift-read
Type: explore

The shifted read takes the previous position's state even when it belongs to the previous packed document.
Zero it at document starts (doc_pos = 0). Tests whether cross-document leakage through the shifted read hurts.
Confirm: < r51d (0.93647).
