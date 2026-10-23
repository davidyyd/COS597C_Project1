# Separately normalized values + low-rank gate in the full-attention layers
Parent: r36b-nolr-ve-sep
Type: exploit (swap: output temperature removed for step time)

r36b (0.93916) has no low-rank self-gate; r35a showed the gate in layers 3, 7 adds ~-0.0007 on the
value-normalized model. Time permitting, add it. Confirm: < r36b.
