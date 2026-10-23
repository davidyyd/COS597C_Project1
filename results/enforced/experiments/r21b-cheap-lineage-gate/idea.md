# Cheap lineage + scalar head self-gate
Parent: r20a-doctemp-mra
Type: exploit

Build the cheaper lineage (doc-temperature + MLP reads attention, 0.95178, 66 ms): add the scalar head
self-gate y_h *= 2 sigmoid(a_h . y_h) (r01b, -0.0024 on baseline, cheap). Confirm: < r20a by > 0.001.
