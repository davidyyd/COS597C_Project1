# d16m6 + attention gate with fused XSA+gate autograd (memory)

Parent: r41d-d16m6-attngate (0.76795, 76.0GB: +3.6GB over r39a). Same math; a custom autograd function for XSA followed by the head gate saves only y, v (already saved by attention), alpha and the gate, recomputing the rest in backward. Goal: gate memory ~free so it fits at d17.
