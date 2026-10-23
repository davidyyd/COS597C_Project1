# Budget swap: scalar self-gate only + prev-token keys
Parent: r14c-chain3-attn-reads-attn
Type: exploit

The rank-32 per-dimension self-gate costs ~4.5 ms (r05b 63.7 -> r08b 68.2 ms) for ~-0.002; the r14c family
is at the step cap. Drop the low-rank part (keep the scalar head self-gate) and re-add prev-token keys
(~-0.0006). Measures how much the low-rank gate is worth in the chained model and frees ~3 ms for other
mechanisms (r18b, r18c). Expect ~ +0.001 vs r14c (0.94455).
