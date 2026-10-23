# Separately normalized values + normalized head output scaled by the self-gate
Parent: r36b-nolr-ve-sep
Type: exploit (swap: output temperature removed for step time)

Head writes norm(y_h) * 2 sigmoid(gate(y_h)) (r36d: -0.0023 vs r01b): direction from attention, magnitude
from the self-gate. Confirm: < r36b (0.93916).
