# Cheap key-shift lineage + MLP reads attention output at t-1
Parent: r54c-r51d-k-shift-only
Type: exploit

MLP reads a(t), a(t-1), m_{l-1}(t) (the read pattern of the best model r56b) on the cheaper lineage without the
head self-gate. Confirm: < r54c.
