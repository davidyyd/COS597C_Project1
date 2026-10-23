# Self-gate + query temperature + multi-scale heads
Parent: r02c-selfgate-qtanh
Type: exploit

Combine the best model (r02c, 0.95353) with multi-scale head windows (r02d, 0.95520 from baseline, -0.0037):
in sliding layers head h has window 1024 >> (3-h) = 128/256/512/1024. Independent mechanisms (where a head
may look vs. how it weights/gates), so gains should add. Confirm: < 0.9520.
