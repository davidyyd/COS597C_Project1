# Notes

Baseline 0.958950 (249 s training). Caps: +5% time (~262 s; 261.66 exactly), params (~52.8M), FLOPs.

## === FINAL STATE (budget used: 400/400) ===
BEST VALID: r100a-r95c-keytemp 0.921311 @ 258.2 s (baseline 0.958950: -0.0376 bpb). Within noise of r95c-r94b-gatesall-ve3
0.921402 (replicate r99a 0.921488 => run noise ~1e-4). Params ~52.8M (cap 52.849M = baseline 50,332,176 x 1.05).
Final stack (r95c + key temp), all on bf16 RoPE tables + bf16 residual (disclosed precision controls):
 - INPUT: token emb + token metadata emb (byte-length x leading-ws x char class) + EXACT DOCUMENT REPETITION
   FEATURES (count of earlier occurrences of x_t in the doc, log2 distance to the last one, n-gram repetition depth
   n<=4 from one batched int32 sort over rolling 16-bit hashes; all causal) -> causal per-channel conv k=4 -> x0
 - stale depth gates on x0 / residual re-injection (read from the previous block's MLP-input norm[:32])
 - layer 0 = causal depthwise conv instead of attention; layer 1 local window 128; multi-res heads (2 of 4 heads
   w=128) in the other sliding layers; full layers 3, 7
 - selective contextual VE (3 lags, per-token lag gates) in layers 1, 5, 7 (VE3 table dropped)
 - per-(head,channel) q gain after QK-norm; selective q/k smears; PER-KEY SALIENCE k *= 2 sigmoid(w.x[:32])
   (shares the smear-gate matmul); POSITION TEMPERATURE for queries and keys (1 + a_{l,h} log(1+t)/log T), folded
   into the RoPE tables (norm first: RMSNorm commutes with the rotation)
 - head + channel attention-output gates; grouped (16) MLP hidden gate in all layers
 - gated mid-network hashed (x_{t-1}, x_t) memory (16384 x 128 -> 512) injected at layer 4 (8-group gate)
 - low-rank n-gram LOGIT PRIORS folded into the head matmul: [x, g*U[x_t], g*U2[x_{t-1}]] @ [W, V, V2]^T, rank
   192 + 96, with 12 context gates (per 24 components) read from the final state; soft-cap 15
Biggest single gains (in the stack): n-gram logit priors (~-0.004), selective ctx-VE, ctx-x0, local early layers,
output gates, q/k smears, VE3->VE5 swap (-0.001), hashed memory (-0.0013), gated priors (-0.0008), repetition
features (-0.0006), position temperature (-3e-4). Control-level (r25a) explore wins of the last phase:
key+value salience -0.0043, doc-mean x0 -0.0030, occurrence features -0.0029, position temperature -0.0024,
MLP token shift -0.0023 (all combined -0.0075: ~60% additive). Most control wins shrink a lot in the stack.
Negatives late: induction candidates (hard successor token/state), cache priors, doc-mean in stack, pointer in
the final stack, adaptive softcap, output-side metadata, lexical salience, value salience in the stack.
Engineering lessons: time cap was binding; Muon timing depends on parameter shapes (e.g. 3 gated MLP layers or
an 8x32 vs 12x32 gate matrix shifted step time by several seconds; r93c's 12 s speedup was such an artifact,
not engineered); fold extra logit terms into the head matmul; fold per-position scales into RoPE tables.

## === STATE after round 69 (historical) ===
BEST VALID: r69a-r68a-split192-96 0.926580 @ 261.1 s (52.7M params: at the param cap; time at the cap).
= r63a (r61b with conv layer 0) + LOW-RANK N-GRAM LOGIT PRIORS folded into the head matmul:
  logits = [x, U[x_t], U2[x_(t-1)]] @ [W_out, V, V2]^T  (rank 192 + 96), with layer-5 VE table dropped
  to fund the rank. Runner-ups: r69c 0.926650 (144+144), r66b 0.927071 (rank 160 bigram only, VE5 kept).
r63a/r61b stack: conv layer 0, local attn layer 1, multires heads, selective ctx-VE (fused lag gates,
2 lags), ctx-x0 (bf16 conv k=4), token metadata embedding, stale depth gates, per-(h,c) q gain,
selective q/k smear, head+channel output gates, grouped MLP hidden gate, U-net skips, bf16 rope/resid.
Key lessons late: prior must be a TABLE (P x0 projection worse); separate output factor per lag;
fold extra logit terms into the head matmul (a separate (B,T,V) add costs ~9 s).
--- older state (after round 50) ---
BEST VALID then: r49b-lean-tokmeta-local0 0.931007 @ 258.8 s.
r49b = r36a-style stack MINUS pointer & readout, PLUS: head+CHANNEL (elementwise) attention-output gates,
grouped (16) low-rank MLP hidden gate, U-net skips, 2-lag ctx-VE, bf16 ctx-x0 conv, TOKEN METADATA
embedding (byte-length bucket x leading-ws x first-char class) added to wte, conv layer 0 -> local attn.
Older description of the r36a stack below (still the core):
Best stack (r36a) = bf16 RoPE tables + bf16 residual stream (disclosed precision controls) +
 - local attention (window 128) in layers 0,1 (instead of conv/1024 windows); multi-res heads (2 of 4 heads
   w=128) in the other sliding layers; full layers 3,7 unchanged
 - contextual x0: causal per-channel conv (k=4) over wte before norm; re-injected via x0 lambdas
 - contextual VE: per-head 4-lag causal conv over each VE lookup (bf16)
 - stale depth gates: per-token x0 gate + residual gate read from previous block's MLP-input norm [:32]
 - per-(head,channel) query gain after QK-norm (replaces per-head temperature)
 - key smear (lag 1, per channel), query smear (lag 1), both SELECTIVE (per-token per-head gate from x[:32])
 - head output gate (per head, from x[:32]), packed gate_bank (Muon per slice); packed attn_scalars
 - depth readout (head reads sum of beta_i * h_i)
 - POINTER-SENTINEL copy head (dense, fp32 scores, 4 causal query chunks): p = a_sent p_vocab + copy
Control for explores: r25a-bf16-resid 0.957260 @ 230.5 s.
Strongest single findings (vs control): ctx-VE -0.009, ctx-x0 -0.0057, head gate -0.0054, local early
layers -0.005, q+k smear -0.0042, pointer -0.0041, temperature/q-gain -0.0027..-0.004, multires heads -0.002.
Negative: hashed bigram emb, VE in keys, lookahead VE, word-so-far emb, bilinear x0, partial RoPE,
multi-base RoPE, value residual, MLP thresholds, x0-as-VE, doc-age, temporal readout, shared VE.
Engineering lessons: see TIMING/COST MODEL below; flex: no learnable tensors in score_mod (50x slow);
flex backward needs grad on its output; flex lowering bugs with score_mods capturing targets -> dense impl;
in-graph doc BlockMask (from_kv_blocks) broke training; mask_mod in loop w/ default arg = wrong masks.
CPU checks: /tmp/gradcheck_ac.py (autocast), /tmp/causal.py, /tmp/ptrcheck.py (pointer causality).

## Tooling
- CPU sanity check: `python /tmp/check.py <train.py>` (forward-only, no_grad; flex has no CPU backward).
  (script: exec the model part of train.py as module "chk", GPT(seq 256), forward). Recreate if /tmp wiped.

## Round 1 (all explore, from baseline)
- r01a-bigram-embed: hashed bigram (prev,cur) embedding added to input, paid by dropping layer-1 VE
- r01b-soft-docmask: learned per-head cross-document logit bias (flex score_mod), all layers flex
- r01c-x0-token-gate: per-token sigmoid gate on x0 re-injection
- r01d-head-temperature: per-head learned temperature after QK-norm

Results R1: a 0.958503 (≈ baseline, bigram hash no help); b TIMEOUT (flex score_mod with a learnable
captured bias -> backward 3.4 s/step vs 66 ms; NEVER put learnable params in score_mod); c 0.954266 (-0.0047);
d 0.954893 (-0.0041). Noise level unknown (single seed) — assume ~0.001-0.002.

## Round 2
- r02a-x0gate-temp (c+d), r02b-depth-gates (c + per-token residual gate), r02c-token-temperature
  (d + per-token temp), r02d-smeared-keys (explore)

Results R2: a 0.951903 (additive! best); b 0.952132 but 265.4 s (> 262 cap — invalid as is);
c 0.955273 (token-dependent temp worse than static: no); d 0.955511 (smeared keys -0.0034).

## Round 3
- r03a-depthgates-temp (r02a + residual gate, fused 32x2 gate, cheaper slice-norm), r03b (r02a + smear),
  r03c-final-x0 (explore: x0 into head), r03d-key-salience (explore: per-key additive logit bias via last channel)

Results R3: a 0.949574 (262.6 s, at the cap 261.7 — borderline); b 0.950055 (269.7 s OVER cap);
c 0.957197 (-0.0018, free); d 0.956250 (-0.0027, ~+1 ms/step).
TIMING: per-step dt baseline ~65-66 ms; cap ≈ 68.6 ms. x0 gate +1ms, resid gate +2-3ms, smear +1-2ms
(fwd+bwd per-token channel reductions on fp32 (B,T,512) are memory-bound). Watch dt in run.log!

## Round 4
- r04a-x0id-gates (r03a, gates from x0 only, one matmul -> faster?), r04b-docmask (hard, known),
  r04c (r02a + salience + head-x0), r04d-logit-confidence (explore: per-token inverse temp on logits)

Results R4: a 0.953449 (66 ms: identity-only gates lose most of the gain => gates need CONTEXT);
b CRASH at eval (trained fine; illegal mem access autotuning compiled create_block_mask under no_grad;
only last 400 log lines kept); c 0.951276 but 72 ms (over cap; salience+hx0 add only -0.0006 to r02a);
d 0.957598 (-0.0014, free).

## Round 5
- r05a-stale-gates (r03a, gates read from prev block's MLP-input norm), r05b-docmask-v2 (eager masks),
  r05c-head-gate (known headwise gated attn), r05d (r02a + head-x0 + conf, free adds)

Results R5: a 0.950453 at 252.8 s (VALID BEST: stale gates keep most of r03a's gain, no time cost);
b 0.956719 docmask works (-0.0022) but eager mask build = 77 ms/step (over cap) -> drop;
c 0.953540 headwise out gate (-0.0054 alone, 67 ms); d 0.953116 (head-side free adds HURT r02a, 70 ms).

## Round 6
- r06a (r05a + head gate), r06b (r05a + smear), r06c-mlp-gate (explore), r06d-rope-head-scale (explore)

Results R6: a 0.948161 (69.9 ms, 265 s OVER); b 0.947613 (70.2 ms, 267 s OVER); c mlp gate 0.957319
(-0.0016, free); d rope per-head scale 0.955631 (-0.0033, but 70 ms?!).
Timing is non-additive & partly unexplained. NOTE: in the baseline q,k reach attention in fp32 (fp32 rope
tables) -> flex S-layers may run fp32. Making that bf16 would speed everything up, but using that to buy
headroom = exploiting the cap -> decided NOT to do it.

## Round 7
- r07a replicate of r06b (noise), r07b stale gates full 512 read, r07c shared VE every layer (explore),
  r07d query-dependent null key via flex LSE (explore)

Results R7: replicate 0.947597 vs 0.947613, 267.3 vs 267.2 s => SAME-SEED NOISE ~2e-5 val, ~0.1 s time:
runs are deterministic; timing costs are real and reproducible. b full-width (512) gate read 0.971479
(much WORSE: high-capacity gate destabilises; keep 32-ch reads); c shared VE every layer 0.960242 (worse);
d query-dependent null key via LSE 0.955263 (-0.0037, < headgate's -0.0054; flex-for-full-layers 69.9 ms).

## Round 8
- r08a (r06a with head gate in bf16), r08b (r05a + mlp gate), r08c (r05a + per-head scalar smear),
  r08d conv replaces layer-0 attention (explore, frees time)

Results R8: a headgate-in-bf16 0.947942 still 266 s (bf16 cast didn't save time); b r05a+mlpgate
0.950175 (256 s, ~no gain over r05a); c per-head scalar smear 0.948893 (worse than per-channel, 267 s);
d CONV LAYER 0 0.955406 at 240.4 s (!!: -0.0035 AND -9 s; layer-0 attention is wasteful).
Gotcha: when a layer drops attention, its attention-only params (log_temp, smear, out_gate) must be set
to None or grads are None -> all_reduce crash. CPU grad check: /tmp/gradcheck.py (fake flex via sdpa).

## Round 9
- r09a conv0+r06b, r09b conv0+r06a, r09c conv0+all, r09d log-n query scaling (explore)

Results R9: a conv0+r06b 0.945345 (250.1 s); b conv0+r06a 0.946651 (248.9 s);
c conv0+stalegates+temp+smear+headgate 0.944515 (258.8 s) = BEST VALID; d log-n query scaling 0.955117
(-0.0038, free).

## Round 10
- r10a r09c+logn, r10b r09c with conv in layers 0,1, r10c r09c+value smear, r10d multi-depth readout (explore)

Results R10: a r09c+logn 0.944439 but 267.5 s (logn free alone, +2.3 ms here!?); b conv in layers 0+1
0.945349 at 245.5 s (+0.0008 val for -13 s: time-for-quality trade); c r09c+value smear 0.943427 but
269.2 s; d depth readout 0.954640 (-0.0043 alone, 259 s).

## Round 11
- r11a conv01 + vsmear (r10c minus layer-1 attention), r11b contextual x0 conv (explore),
  r11c/r11d DIAGNOSTIC profiler runs of r09c / r10a (to understand the non-additive time costs)

Results R11: a conv01+vsmear 0.944633 (250.3 s); b CONTEXTUAL x0 (causal per-channel conv k=4 on wte
before norm) 0.953263 (-0.0057 alone, 252.7 s!); c/d diagnostics.
PROFILE (r11c): per step ~fwdbwd 50 ms, grad all-reduce 5.6 ms, optimizer 14.4 ms. ~82 small NCCL
all-reduces/step at ~0.15 ms each: EVERY EXTRA PARAMETER TENSOR COSTS ~0.15 ms/step (~0.6 s train time).
=> pack per-layer extras into GPT-level tensors (attn_scalars 1-D AdamW; gate_bank.attn.out_gate
(L,H,32) -> Muon per slice, name must contain ".attn."). Same math, far cheaper.

## Round 12
- r12a packed r10c, r12b packed+logn+ctx-x0, r12c packed+readout+ctx-x0, r12d doc-age feature (explore)

Results R12: a packed r10c 0.943630 at 265.7 s (packing saved only 3.5 s: tensor count is NOT the main
cost); b +logn+ctx-x0 0.943865 (no gain, 269.7); c +readout+ctx-x0 0.941288 (BEST VAL, but 274.4 s);
d doc-age 0.956900 (-0.002, meh).
COST MODEL (revised): each elementwise pass over an fp32 (B,T,512) tensor ≈ 0.05 ms (bf16 ≈ 0.025);
residual stream/x0/h are fp32 -> per-layer ops on them (readout, gates, smears on fp32 v) add up.
Conv layer replaces attention: -2.3 ms (layer 0), -3.5 ms (layer 1).

## Round 13
- r13a r12c+conv layer1, r13b r12c+conv1-vsmear, r13c r12c fused bf16 readout, r13d contextual VE (explore)

Results R13: a r12c+conv1 0.943328 (267 s; conv1 costs +0.002 here, saves 7 s); b r12c+conv1-vsmear
0.943643 at 258.2 s (VALID); c fused bf16 readout 0.941295 at 271.7 (saves 2.7 s only);
d CONTEXTUAL VE (conv k=4 on each VE lookup) 0.949978 alone (-0.009!!) but 293.6 s (fp32).

## Round 14
- r14a r13b + ctx-VE (bf16), r14b ctx-VE bf16 alone, r14c/d diagnostic profiles of r13d / r12c

Results R14: a r13b + ctx-VE(bf16) 0.938861 at 264.7 s (3 s OVER; ctx-VE cost only +6.5 s here);
b ctx-VE bf16 alone 0.950333 at 288.7 s; c/d profiles: r13d has ONE pathological Inductor kernel
(triton_per_fused__fused_rms_norm_backward..._constant_pad...) = 8.4 ms/step: the per-channel conv
weight-grad (column reduction over B*T) got fused into a persistent-reduction kernel. Fusion is
model-dependent (absent in r14a). Muon step ~13.7 ms, all-gather ~8 ms, grad all-reduce ~4-10 ms/step.

## Round 15
- r15a r14a - key smear, r15b r14a - readout, r15c r14a per-head VE conv weights, r15d multi-base RoPE (explore)

Results R15: a r14a-keysmear 0.941889 (255 s: key smear still worth 0.003!); b r14a-readout 0.939664
at 251.8 s = BEST VALID; c per-head VE conv 0.938485 (better than per-channel; 266 s over);
d multi-base RoPE per head 0.959510 (worse than baseline: negative).

## Round 16
- r16a r15b + per-head VE conv, r16b r15b with attention back in layer 1, r16c r15b + per-head 4-lag
  value conv, r16d temporal readout (explore)

Results R16: a r15b + per-head VE conv 0.939117 at 252.7 s = BEST VALID; b attention back in layer 1
0.938104 but 269.3 s (layer-1 attn worth -0.001, costs 17 s); c per-head 4-lag value conv 0.939453 (no
gain, +16 s: in-layer convs are slow); d temporal readout 0.958043 (-0.0009 only, +9 s: negative).

## Round 17
- r17a layer 1 = local attention window 128, r17b r16a + bilinear e_t*e_{t-1} in x0, r17c r16b - head gate,
  r17d bilinear x0 on r11b (explore)

Results R17: a layer 1 = local attention (window 128) 0.937013 (BEST VAL; 266.5 s over: window size
doesn't save time -> the per-attention-layer overhead is the fp32 q/k path + extras); b bilinear
e_t*e_{t-1} in x0 0.938938 (~0); c r16b - head gate 0.939234 (head gate worth ~0.001 there);
d bilinear on r11b 0.952978 (~0: NEGATIVE result, bigram identity not useful as such).

## Round 18
- r18a baseline with bf16 RoPE tables (precision/efficiency control: q,k fp32 in baseline flex layers),
  r18b r17a+bf16 rope, r18c r16b+bf16 rope, r18d local early layers 0-1 window 128 (explore)

Results R18: a baseline + bf16 RoPE tables 0.957390 at 241.9 s (-0.0016 AND -7 s: fp32 attention in
the baseline was an accident; disclosed control); b r17a + bf16 rope 0.936916 at 261.0 s = BEST VALID
(just under 261.66 cap); c r16b + bf16 0.938138 (264.5 over); d LOCAL EARLY layers 0-1 window 128 on
baseline 0.954016 (-0.0049, 245.8 s) > conv0 (-0.0035).

## Round 19
- r19a r18b with layer 0 local attn instead of conv, r19b r18b + layer 2 local, r19c r18b + value smear
  (bf16 now), r19d selective (token-gated lags) contextual x0 on r11b (explore)

Results R19: a local attn layer 0 (no conv) 0.936355 (271 s over); b layer 2 local 0.937719 (worse,
253 s); c + value smear 0.936320 (268.6 s over); d selective lag gates on ctx-x0 (r11b) 0.951505 (-0.0018).
From now on explores start from r18a (bf16-rope baseline, 0.957390 @ 241.9 s) as the control.

## Round 20
- r20a/b diagnostic profiles of r18b / r19c, r20c MLP thresholds (explore, on r18a), r20d r18b + selective ctx-x0

Results R20: profiles: r18b fwdbwd 49.7 ms, grad all-reduce 5.4, optimizer (Muon) 14.3 ms/step;
value smear costs +1.7 ms fwdbwd (real compute; ~0.2 ms/layer per extra elementwise op on (B,T,512)).
Muon step is ~20% of step time (7 rounds of 8 params; all-gather ~7-8 ms). c MLP per-unit thresholds
0.957657 (worse than r18a: NEGATIVE); d selective lags in r18b 0.937027 (no gain).
Headroom left in r18b: ~0.7 s. => need zero-cost ideas or swaps.

## Round 21
- r21a r18b + ctx-VE into keys, r21b VE into keys on r18a (explore), r21c ctx-x0 width 8, r21d ctx-x0 -> head

Results R21: a ctx-VE into keys 0.936864 (~0, 264 s); b VE into keys on r18a 0.957875 (worse: NEGATIVE);
c ctx-x0 width 8 0.936697 (-0.0002, +4.3 s); d ctx-x0 skip to head 0.936758 (-0.0002, +5.4 s).
Even once-per-forward fp32 ops cost ~1 ms/step. Plateau ~0.9369 valid.

## Round 22 (ablations of r18b to price components)
- r22a -headgate, r22b -stale depth gates, r22c -temperature; r22d x0 as VE in non-VE layers (explore)

Results R22 (ablations of r18b 0.936916 @ 261.0): -headgate 0.938196 @ 250.9 (worth 0.0013 / 10 s);
-depth gates 0.939238 @ 251.6 (0.0023 / 9.4 s); -temperature 0.938760 @ 254.2 (0.0018 / 6.8 s).
=> all components more time-efficient than the candidates tried (local0 0.0006/10 s, vsmear 0.0006/7.6 s).
d x0 as VE in non-VE layers 0.958303 (worse than r18a: NEGATIVE).

## Round 23: LOOKAHEAD VALUE EMBEDDINGS (new idea)
Keys s<t also offer VE(token s+1) as extra value; exactly causal via strictly-causal flex pass renormalised
with LSEs: y = y_causal(v) + exp(lse_strict - lse_causal) * y_strict(n). Key 0 kept in strict mask so no
empty rows; row 0 term zeroed. CPU causality test (/tmp/causal.py: perturb token t+1, diff at <=t) = 0.
- r23a on r18a layer 7, r23b layers 3+7, r23c r18b + L7, r23d r22a (no headgate) + L7

Results R23: lookahead VE NEGATIVE: r23a 0.958687 (worse than r18a 0.95739), r23b 0.958218, r23c in
stack 0.936719 (-0.0002, 272 s), r23d 0.938420.

## Round 24
- r24a r18b + doc masking (in-graph BlockMask.from_kv_blocks; verified vs dense: /tmp/maskcheck.py),
  r24b same on r18a, r24c word-so-far embedding (explore), r24d r18b + word-so-far
GOTCHA: a mask_mod defined in a loop with a default arg (w=w) gave WRONG masks (flex reuses by code?);
use a factory closure (like baseline _mask). Existing sliding masks verified OK (/tmp/maskcheck2.py).

Results R24: r24a/b doc masking with in-graph from_kv_blocks CRASHED (losses far too high from step 100,
NaN by ~700; CPU forward matched dense ref -> backward/compile issue). Dropped doc masking.
c word-so-far embedding 0.957276 (≈ r18a: NEGATIVE); d in r18b 0.937207 (worse).

## Round 25: bf16 residual stream (disclosed precision/efficiency control)
- r25a r18a+bf16 resid, r25b r18b+bf16, r25c r19c(vsmear)+bf16, r25d r19a(local0)+bf16

Results R25: bf16 residual: r25a 0.957260 @ 230.5 s (control: ≈ r18a, -11 s); r25b r18b+bf16 0.937237
@ 242.4 (+0.0003, -18.6 s); r25c vsmear+bf16 0.936333 @ 247.7; r25d local0+bf16 0.936258 @ 247.8 = BEST VALID.
New control for explores: r25a (0.957260 @ 230.5 s). CPU grad check for bf16-resid models needs autocast:
/tmp/gradcheck_ac.py.

## Round 26
- r26a local0+vsmear+bf16, r26b + readout, r26c + ctx8 + head ctx skip, r26d multi-resolution heads
  (2 local heads w=128 + 2 long heads per S layer; explore on r25a)

Results R26: a local0+vsmear+bf16 0.935998 @ 256.8 = BEST VALID; b +readout 0.935638 @ 266.6 (over);
c +ctx8+headctx 0.935782 @ 268.6 (over); d MULTI-RES HEADS (2 of 4 heads w=128 in S layers) 0.955269
@ 223.9 (-0.002 vs r25a AND -6.6 s).

## Round 27
- r27a r26a + multires heads, r27b + readout, r27c + readout + ctx8 + headctx, r27d multires in all layers (explore)

Results R27: a r26a+multires 0.936280 @ 249.4 (+0.0003, -7.4 s); b +readout 0.935842 @ 252.3 = BEST
VALID; c +ctx8+headctx 0.935722 @ 267.1 (over); d multires in all layers 0.957164 (worse than S-only).

## Round 28
- r28a per-channel q gain in QK-norm (RoPE-band weighting; explore), r28b partial RoPE (known),
  r28c r27b + q gain, r28d value residual (ResFormer, known) on r25a

Results R28: a per-channel q gain (RMSNorm weight) on r25a 0.954566 (-0.0027, free); b partial RoPE
0.960115 (worse: NEGATIVE); c r27b + packed (1+g) q gain 0.938814 (+0.003 WORSE: conflicts w/ temp?);
d value residual (ResFormer) 0.957866 (worse: NEGATIVE with VEs).

## Round 29
- r29a r27b temp -> per-(head,channel) gain, r29b r27b + separate RMSNorm-weight q gain, r29c MLP input
  token shift (explore), r29d per-(head,channel) gain on r25a

Results R29: a r27b temp -> per-(head,channel) q gain (packed, 1+g) 0.935361 @ 252.6 = BEST VALID;
b r27b + separate RMSNorm-weight q gain (keeps temp) 0.935808 (~0); c MLP INPUT TOKEN SHIFT on r25a
0.954864 (-0.0024, +4 s); d per-(head,channel) gain on r25a 0.954770 (-0.0025).

## Round 30
- r30a r29a + MLP token shift, r30b r30a - key smear, r30c attention-input token shift (explore),
  r30d r30a + per-layer plain/ctx x0 mix

Results R30: a r29a + MLP token shift 0.935303 @ 264.1 (~0, +11.5 s); b r30a - key smear 0.938107
(KEY SMEAR WORTH 0.003 in stack!); c attention-input token shift on r25a 0.955424 (-0.0018);
d r30a + plain/ctx x0 mix 0.934988 @ 271.1 (over).

## Round 31 (induction-oriented: bigram queries/keys)
- r31a r29a + lag-2 key smear, r31b r29a + query smear, r31c both, r31d q+k smear on r25a (explore)

Results R31: a lag-2 key smear 0.935133 @ 265 (over); b QUERY SMEAR 0.934373 @ 266.5 (-0.001, over);
c both 0.934142 @ 269.9 (over); d q+k smear on r25a 0.953065 (-0.0042).

## Round 32
- r32a r31b - vsmear, r32b r31c - vsmear (fit the cap), r32c POINTER-SENTINEL copy head on r25a
  (explore; exact via 2 one-head flex passes with LSEs; causal-checked /tmp/ptrcheck.py), r32d r29a + pointer

Results R32: a r31b - vsmear 0.934967 @ 252.6 = BEST VALID; b r31c - vsmear 0.934386 @ 264 (over);
c/d pointer CRASHED: Inductor lowering of flex_attention_backward fails if the flex OUTPUT has no grad
(only lse used) -> tie the outputs in with 0.0 * o.

## Round 33
- r33a r32b - readout (fit lag-2 key smear), r33b r32a - readout (ablation), r33c pointer v2 on r25a,
  r33d r32a + pointer

Results R33: a r32b - readout 0.935171 @ 252.3; b r32a - readout 0.935406 @ 248.8 (readout worth
0.0004 / 3.8 s); c/d flex pointer v2 CRASHED again (Inductor lowering "'function' object has no attribute
'graph'" in flex_attention) -> switched to a dense (B,T,T) implementation.

## Round 34
- r34a r32b - residual gate, r34b selective (token-gated) q/k smear on r31d (explore),
  r34c DENSE pointer on r25a (explore), r34d r32a + dense pointer

Results R34: a r32b - resid gate 0.935632 @ 260.4 (resid gate worth ~0.0012); b SELECTIVE q/k smear on
r31d 0.951925 (-0.0011); c DENSE POINTER on r25a 0.953150 (-0.0041 vs r25a, +6 s) WORKS;
d r32a + dense pointer 0.934204 @ 265 (-0.0008, over by 3.4 s).

## Round 35
- r35a r34d with bf16 pointer scores, r35b r35a - readout, r35c pointer keys from x0 (explore),
  r35d r32a + selective q/k smear

Results R35: a pointer w/ bf16 scores 0.934772 @ 264.8 (bf16 S HURTS, no speedup); b r35a - readout
0.934600 @ 256.9; c pointer keys from x0 0.956725 (worse); d r32a + SELECTIVE q/k smear 0.934208 @ 254.9
= BEST VALID.

## Round 36
- r36a r35d + chunked fp32 pointer, r36b r36a - readout, r36c pointer + exact n-gram match priors
  (explore), r36d chunked pointer on r25a (reference)

Results R36: a r35d + chunked pointer 0.933890 @ 261.5 = BEST VALID; b - readout 0.934458 @ 258.6;
c pointer + n-gram match priors 0.952853 (vs d 0.953060: -0.0002 only); d chunked pointer on r25a
0.953060 @ 236.0.

Results R37: a profiled r36a 0.934486 (vs 0.933890 unprofiled: RUN NOISE ~5e-4 for pointer models!
small gains <5e-4 are not reliable); b 8 pointer chunks 0.933875 @ 261.6 (no time saved); c 2-head
pointer 0.953374 (worse); d pointer recency bias 0.953223 (worse). Profile: fwdbwd 51 ms, Muon 12 ms,
grad all-reduce 5.6 ms; no pathological kernels.

## Round 38
- r38a r36b + selective lag-2 key smear, r38b r36a with 32-dim pointer, r38c U-net skips (known, explore),
  r38d elementwise low-rank attention-output gate (explore)

Results R38: a r36b + selective lag-2 key smear 0.934428 @ 260.3 (no gain); b r36a w/ 32-dim pointer
0.933887 @ 260.9 (= r36a, slightly faster; new base); c U-net skips on r25a 0.956756 (-0.0005);
d ELEMENTWISE low-rank attn-output gate on r25a 0.951943 (-0.0053, +9.7 s).

## Round 39
- r39a r38b headgate -> elementwise gate, r39b both gates, r39c low-rank MLP hidden gate (explore),
  r39d per-channel VE gate (explore)

Results R39: a r38b headgate->ELEMENTWISE gate 0.932986 @ 264.0 (-0.0009, over); b BOTH head+channel
gates 0.932123 @ 268.7 (-0.0018, over 7 s); c low-rank MLP hidden gate on r25a 0.953909 (-0.0034, +18.5 s);
d per-channel VE gate 0.957751 (worse; oddly 196 s).

## Round 40
- r40a r39b - pointer - readout, r40b r39b - pointer, r40c r39a - readout, r40d low-rank MLP *input*
  gate (explore)

Results R40: a r39b - pointer - readout 0.932561 @ 260.4 = BEST VALID; b r39b - pointer 0.932412 @ 264.2
(over); c r39a - readout 0.933455 @ 261.1; d low-rank MLP INPUT gate on r25a 0.955074 (-0.0022, +11 s).
(=> with output gates, the pointer is worth ~0.0003 for 4.5 s: dropped.)

## Round 41 (pricing r40a's components)
- r41a - selective smear gate, r41b - ctx-VE conv, r41c - q gain; r41d grouped MLP hidden gate (explore)

Results R41 (ablations of r40a 0.932561 @ 260.4): -selective smear gate 0.933383 @ 264.6 (worth 0.0008,
FREE); -ctx-VE conv 0.937652 @ 250.2 (worth 0.0051 / 10 s); -q gain 0.934106 @ 261.8 (worth 0.0015, FREE);
d grouped MLP hidden gate (128 gates) on r25a 0.954811 (-0.0024, +5.4 s).
Compile/timing variance: removing a component can make a run slower (+-3 s).

## Round 42
- r42a r40a + per-channel resid/x0 lambdas, r42b r40a w/ conv layer 0 + grouped MLP gate, r42c r40a +
  grouped MLP gate, r42d elementwise MLP-output gate (explore)

Results R42: a per-channel resid/x0 lambdas 0.933208 (worse); b r40a w/ conv layer 0 + GROUPED MLP
hidden gate 0.932459 @ 252.7 = BEST VALID (9 s headroom); c r40a + grouped MLP gate 0.931985 @ 262.9
(over by 1.2 s); d elementwise MLP-output gate on r25a 0.955147 (-0.0021, +7.5 s).

## Round 43
- r43a r42b + readout, r43b r42b + pointer(32), r43c both, r43d gates reading a learned rank-32
  projection of the full input (explore)

Results R43: a r42b + readout 0.932089 @ 262.0 (0.34 s OVER); b + pointer 0.932017 @ 263.7 (over);
c both 0.931876 @ 266.5 (over); d gates reading a learned rank-32 full-input projection 0.951858 (≈ r38d: no gain).

## Round 44
- r44a r42b + readout of layers 4-7, r44b r42b + MLP-output gate, r44c r42b + U-net skips,
  r44d key-side value gate (explore)

Results R44: a r42b + late readout 0.932382 @ 260.8; b + MLP-out gate 0.932255 @ 265.1 (over);
c r42b + U-NET skips 0.932038 @ 258.9 = BEST VALID; d key-side value gate on r25a 0.955896 (-0.0014).

## Round 45
- r45a r44c + late readout, r45b r44c + doc-so-far mean in x0, r45c doc-so-far mean on r25a (explore),
  r45d r44c + key-side value gate

Results R45 (all ≈ r44c within noise): a + late readout 0.932181 @ 261.9 (over); b + doc-so-far mean
0.932101 @ 261.3; c doc-so-far mean on r25a 0.957179 (≈ control: NEGATIVE); d + key-side v gate 0.932495
@ 264.7 (over). Params: r44c 50.0M (cap 52.8M).

## Round 46
- r46a ctx-VE 2 lags (cost), r46b ctx-x0 conv in bf16 (cost), r46c selective ctx-VE lags, r46d per-token
  learned embedding scale (explore)

Results R46: a ctx-VE 2 lags 0.932192 @ 248.3 (≈, -10.6 s!); b ctx-x0 conv bf16 0.932147 @ 254.1 (≈, -4.8 s);
c selective ctx-VE lags 0.931433 @ 287.4 (-0.0006 but +28 s: pathological kernel); d per-token embedding
scale on r25a 0.956515 (-0.0007).

## Round 47
- r47a lean base (r44c + 2-lag ctx-VE + bf16 ctx-x0), r47b + local layer 0, r47c + MLP-out gate + late
  readout, r47d value RoPE (explore)

Results R47: a LEAN base (r44c + 2-lag ctx-VE + bf16 ctx-x0) 0.932349 @ 246.8 (15 s headroom);
b + local layer 0 0.932038 @ 261.5 (= r44c); c + MLP-out gate + late readout 0.932385 @ 264.6 (over,
no gain); d value RoPE on r25a 0.961836 (WORSE: negative).

## Round 48
- r48a lean + per-unit MLP hidden gate, r48b lean + writable x0 stream, r48c lean + key-side v gate,
  r48d token metadata embedding (explore)

Results R48: a lean + per-unit MLP gate 0.931732 @ 271.6 (-0.0006, over by 10 s); b writable x0 stream
0.932601 (worse); c key-side v gate 0.932611 (worse); d TOKEN METADATA embedding on r25a 0.955707
(-0.0016, free).

## Round 49
- r49a lean + token meta, r49b + local layer 0, r49c + MLP gate groups of 8, r49d char 3-gram hashed
  embeddings (FastText-style, 4096 buckets, embedding_bag per vocab entry; explore)

Results R49: a lean + TOKEN META 0.931473 @ 247.2; b + local layer 0 0.931007 @ 258.8 = BEST VALID;
c MLP gate groups of 8 0.931581 (no gain); d char 3-gram hashed embeddings on r25a 0.956130 (-0.0011, +10 s).

## Round 50
- r50a r49b + last-char class in meta, r50b r49a + char n-grams, r50c r49b + meta in VE tables,
  r50d position-in-word embedding (explore, bf16 control)

Results R50 (≈ r49b within noise): a + last-char class 0.931021 @ 261.2; b r49a + char n-grams
0.931250 @ 264.3 (over); c meta in VE tables 0.931352 (worse); d POSITION-IN-WORD embedding on r25a
0.955738 (-0.0015, free).

## Round 51
- r51a r49b + position-in-word, r51b r49a + pointer, r51c r49b + dilated ctx-x0 lags (4,8,12),
  r51d sentence-position embedding (explore)

Results R51: a r49b + pos-in-word 0.930970 (≈); b r49a + POINTER 0.930749 @ 260.9 = BEST VALID;
c dilated ctx-x0 lags 0.931024 (≈); d sentence-position embedding on r25a 0.956286 (-0.001, free).

## Round 52
- r52a r51b + doc masking in the 2 full layers (static causal block mask + non-learnable score_mod),
  r52b differential attention in full layers on r25a (explore, known), r52c r51b + pos-in-word,
  r52d r51b + sentence position

Results R52: a r51b + DOC MASKING in the 2 full layers (static causal block mask + score_mod with captured
doc ids) 0.929378 @ 265.0 (-0.0014, over by 3.3 s) — works where the in-graph BlockMask approach failed;
b diff-attn CRASH (lambda params in non-full layers had no grad; gradcheck misses this because its fake
masks route all layers through the full path); c + pos-in-word 0.930676 (≈); d + sentence pos 0.930845 (≈).

## Round 53
- r53a r49a + doc mask full layers, r53b r49a + doc mask in all layers, r53c r53a + pos-in-word,
  r53d diff attention v2 (explore)

Results R53: a r49a + doc mask full 0.931646 @ 257.8 (NO gain on r49a!); b doc mask all layers
0.931225 @ 260.2 (-0.00025); c + pos-in-word 0.931564; d diff-attn full layers on r25a 0.958485 (worse).
=> r52a's -0.0014 may be partly noise. Round 54 measures noise with replicates.

## Round 54
- r54a replicate r52a, r54b replicate r51b, r54c r52a - U-net (fit cap), r54d cross-layer KV sharing
  (2<-1, 6<-5) on r25a (explore, time saver)

Results R54: REPLICATES: r52a 0.929378 -> 0.929183, r51b 0.930749 -> 0.930789 => noise ~2e-4;
doc masking + pointer gain (-0.0015) is REAL (but doc masking without pointer ~0: interaction!).
c r52a - U-net 0.929584 @ 264.7 (no time saved); d cross-layer KV sharing on r25a 0.963929 (much worse).

## Round 55
- r55a r52a - MLP gate, r55b r52a head-gate only, r55c r55a + pointer restricted to same doc,
  r55d fixed ALiBi slopes in full layers (explore)

Results R55: a r52a - MLP gate 0.930372 @ 261.6 = BEST VALID (MLP gate worth ~0.001 here!);
b head gate only 0.931574 @ 253 (channel gate worth 0.0022); c pointer restricted to same doc 0.930666
(worse: cross-doc copying useful); d fixed ALiBi in full layers 0.963375 (much worse).

## Round 56
- r56a r55a with SDPA bool-mask doc masking (speed), r56b r52a with 1-lag ctx-VE (fit), r56c profile r52a,
  r56d fixed induction prior in full layers (explore)

Results R56: a SDPA bool-mask doc masking 0.930378 @ 274.2 (slower than flex score_mod); b r52a w/ 1-lag
ctx-VE 0.930051 @ 263.2 (over 1.5 s); c profile r52a: doc-masked full-layer flex bwd ~1.26 ms/layer
(2x an S layer); d induction prior CRASH: Inductor flex lowering bug ("'function' object has no attribute
'graph'") when a score_mod captures TWO index tensors (one captured tensor works) -> pack into one.

## Round 57
- r57a r52a + conv layers 0-1, r57b r52a + 3 local heads, r57c r56b + conv 0-1, r57d induction prior v2

Results R57: a r52a + conv layers 0-1 0.934439 (+0.005: layer-1 attention essential now); b 3 local heads
0.930063 @ 262.8 (over 1.1 s); c r56b + conv01 0.934336 (bad); d induction prior v2 CRASH again (same
lowering error) -> the real trigger is the score_mod FORM: `score + where(cond, c1, c2)` fails to lower;
`where(cond, score, c)` works (doc mask, ALiBi arithmetic also fine).
FLEX POINTER rewritten in the safe form: CPU check vs dense pointer: same NLL up to bf16 (/tmp/ptrcompare.py).

## Round 58
- r58a r52a w/ flex pointer, r58b r55a w/ flex pointer, r58c r52a 1-lag + 3 local heads, r58d induction prior v3

Results R58: a/b flex pointer (safe form) CRASHED again with the same lowering error; d induction prior v3
CRASHED too => score_mods indexing token-id tensors (idx/targets-derived) fail to lower; only the doc-id
mask and arithmetic (ALiBi) score_mods work. Keep the dense pointer. c r52a 1-lag + 3 local heads
0.930374 @ 260.7 (valid, = r55a).

## Round 59
- r59a r52a doc mask only layer 7 + 3 local heads, r59b r55a + pos-in-word + sentence pos,
  r59c r55a + per-head selective ctx-VE lags, r59d induction prior via SDPA additive mask (explore)

Results R59: a doc mask only layer 7 + 3 local 0.930701 @ 260.4 (worse); b + pos-in-word + sent pos
0.931198 (no gain); c r55a + PER-HEAD SELECTIVE ctx-VE lags 0.929398 @ 265.6 (-0.001, over 4 s);
d induction prior (SDPA additive mask) on r25a 0.959545 (worse: negative).

## Round 60
- r60a r59c channel gate only, r60b r59c + 3 local heads, r60c r49b + selective ctx-VE, r60d data-dependent
  key scale (explore)

Results R60 (all over by 1-2 s): a r59c channel gate only 0.930745 @ 262.6; b r59c + 3 local 0.930338
@ 263.8; c r49b + SELECTIVE ctx-VE 0.930155 @ 262.5; d data-dependent key scale on r25a 0.955928 (-0.0013).

## Round 61
- r61a r60c - VE gate, r61b r60c fused lag-gate matmul, r61c r60c - U-net, r61d content-dependent RoPE
  position offsets (explore)

Results R61: a r60c - VE gate 0.931201 @ 261.3 (VE gate worth 0.001); b r60c fused lag-gate matmul
0.929948 @ 262.1 (over by 0.4 s!); c r60c - U-net 0.930225 @ 262.2; d content-dependent RoPE offsets on
r25a 0.957587 (-0.0004, weak).

## Round 62
- r62a r61b - U-net, r62b r61b ctx-x0 k=2, r62c r62a + key scale (via smear gate projection),
  r62d normalised residual stream (explore)

Results R62: a r61b - U-net 0.930149 @ 262.2; b ctx-x0 k=2 0.930211 @ 262.1 (both 0.5 s over);
c + key scale 0.931708 @ 243.5 (worse val; oddly 19 s FASTER: param-shape changes alter Muon's per-round
all-gather/broadcast layout — optimizer-implementation artifact, not exploited deliberately);
d normalised residual stream 0.978976 (much worse: NEGATIVE).

## Round 63
- r63a r61b w/ conv layer 0, r63b r63a + MLP-out gate, r63c r61b + 3 local heads, r63d low-rank bigram
  logit prior (explore)

Results R63: a r61b w/ conv layer 0 0.930675 @ 253.7 (valid, 8 s headroom); b + MLP-out gate 0.931151 @
263 (worse); c 3 local heads 0.931059 @ 260.5; d LOW-RANK BIGRAM LOGIT PRIOR (logits += U[x_t] V^T, rank 64)
on r25a 0.954901 (-0.0024, +5 s).

## Round 64
- r64a r63a + bigram prior, r64b r49a + fused selective ctx-VE + bigram prior, r64c r63a + rank-128 prior,
  r64d skip-bigram prior (explore)

Results R64: BIGRAM PRIOR IN STACK: a r63a + rank-64 0.928542 @ 263.0 (-0.0021, over 1.3 s); b r49a +
sel-ctxVE + bigram 0.928886 @ 262.9; c r63a + RANK-128 0.927758 @ 263.6 (-0.0029, over 2 s); d skip-bigram
(prev token factor too) 0.955346 (worse than plain).
The separate (B,T,8192) logits add costs ~9 s -> fold into the head matmul: [x, U[x_t]] @ [W_out, V]^T.

## Round 65
- r65a r64c fused (rank 128), r65b r64a fused (rank 64), r65c r49a+selVE+bigram128 fused, r65d gated bigram (explore)

Results R65: FUSED bigram prior: a r63a + rank-128 fused 0.927801 @ 260.2 = BEST VALID (-0.0026 vs r55a);
b rank 64 fused 0.928755 @ 258.9; c r49a + selVE + rank-128 fused 0.927900 @ 260.0; d context-gated prior
on r25a 0.955123 (worse than ungated). Params: rank 128 -> 52.2M (cap 52.815M).

## Round 66
- r66a prior input = P x0_ctx (contextual n-gram prior, rank 128), r66b rank 160, r66c ctx prior rank 256,
  r66d ctx (lags 0-3) prior on control (explore)

Results R66: a prior input = P x0_ctx (rank 128) 0.929534 (worse: the U TABLE matters); b RANK 160 table
0.927071 @ 261.5 = BEST VALID (52.71M params, at the param cap); c ctx prior rank 256 0.928585 @ 263.4;
d ctx (lags 0-3) prior on r25a 0.958076 (worse than bigram table).
FLOPs note: prior adds 2*r*8192 FLOPs/token (rank 160: +2.6M, ~3.7%) but conv0/local windows/multires
heads save more attention FLOPs, so the model stays under the FLOPs cap.

## Round 67
- r67a/b trade layer-1 / layer-5 VE table for bigram rank 416, r67c drop layer-1 VE + rank 288,
  r67d hashed trigram-context prior table (explore)

Results R67: a drop VE1 + rank 416 0.928263 (worse: layer-1 VE matters); b DROP VE5 + RANK 416 0.926810
@ 263.9 (-0.0003, over 2.2 s); c drop VE1 + rank 288 0.928642; d hashed trigram prior table on r25a 0.955423
(worse than bigram table).

## Round 68
- r68a drop VE5 + rank 288, r68b drop VE5 + rank 416 + 3 local heads, r68c drop VE5 + rank 352,
  r68d skip-bigram with separate output factor (explore)

Results R68: a drop VE5 + rank 288 0.927213 @ 261.2 (valid, ≈ r66b); b rank 416 + 3 local 0.927059 @ 262.1
(over); c rank 352 0.926957 @ 262.3 (over); d SKIP-BIGRAM with its own output factor on r25a 0.953907
(-0.001 vs the plain bigram prior r63d).

## Round 69
- split prior rank between current and previous token: r69a r68a 192+96, r69b r66b 112+48, r69c r68a 144+144;
  r69d decomposed trigram (t, t-1, t-2; rank 48 each) on r25a (explore)

Results R69: a SPLIT 192 (x_t) + 96 (x_(t-1)) 0.926580 @ 261.1 = BEST VALID; b r66b split 112+48 0.927277
@ 263 (over); c split 144+144 0.926650 @ 261.1; d decomposed trigram (t, t-1, t-2) on r25a 0.954188 (worse
than skip-bigram).
Results R70: replicate of r69a 0.926560 @ 261.2 (robust); a + unigram bias 0.926555 @ 262.6 (no gain, over);
c position-in-word logit prior 0.957625 & d document-position logit prior 0.957700 on r25a (NEGATIVE).

## Round 71
- r71a shared U across lags (rank 160 each), r71b split 224+64, r71c drop VE3 instead of VE5,
  r71d bigram prior added after the soft-cap (explore)

Results R71: a shared U across lags 0.927121 (worse); b split 224+64 0.927034 (worse); c DROP VE3 (instead
of VE5) 0.925878 @ 262.2 (-0.0007, over by 0.5 s); d prior after soft-cap on r25a 0.954757 (-0.00015, tiny).

## Round 72
- r72a drop VE3+VE5 (192+96), r72b drop VE3+VE5 + rank 224+128, r72c r71c with 176+96,
  r72d factorised (rank-256) VE tables on r25a (explore: parameter source)

Results R72: a drop VE3+VE5 0.927609 @ 258.4 (VE5 worth ~0.0017 once VE3 is gone); b + rank 224+128
0.926689 @ 259.8; c r71c 176+96 0.926212 @ 262.1 (over); d FACTORISED VE (rank-256 tables + up-proj) on r25a
0.956474 (-0.0008 vs control with 8.4M FEWER params!).

## Round 73
- r73a r71c + factorised VE, r73b r69a with all 4 VE layers factorised, r73c r73b + prior 256+128,
  r73d factorised VE in all 8 layers (explore)

Results R73: factorised VE IN THE STACK HURTS: a r71c + fact VE 0.931428 @ 263.4; b all-4 fact VE 0.930315
@ 266.3; c + rank 384 0.929157 @ 267.7 (all worse & slower: clashes with ctx-VE conv / lag gates?);
d FACTORISED VE IN ALL 8 LAYERS on r25a 0.953385 (-0.0039 vs control, +11 s).

## Round 74
- r74a r71c - U-net, r74b r71c - MLP gate (fit the cap); r74c fact VE all 8 + ctx conv on 256-d lookups,
  r74d fact VE all 8 at rank 128 (explores)

Results R74: a r71c - U-net 0.926874 @ 261.7 (0.04 s over!); b r71c - MLP gate 0.927374 @ 257.8;
c fact VE all 8 + ctx conv on r25a 0.951842 (-0.0054, +15 s); d fact VE all 8 rank 128 0.957770 (worse).
NOTE: r69a has 50.6M params (not 52.7M): ~2.2M param headroom; time is the binding cap.

## Round 75: INDUCTION CANDIDATE (new): per-row stable sort finds the most recent earlier occurrence of
x_t; its successor token (position <= t, causal; CPU check diff=0) = induction candidate.
- r75a r69a with prior re-split 160+64+64(induction), r75b r69a + induction-candidate input embedding,
  r75c induction logit prior on r25a, r75d induction input embedding on r25a

Results R75: INDUCTION CANDIDATE IS NEGATIVE everywhere: a prior 0.927022 (vs 0.92658), b embed 0.927646,
c prior on r25a 0.957328 (~0), d embed on r25a 0.957170 (~0). Attention already does induction well.

## Round 76 (100 left): price r69a components (val lost vs seconds saved) to fund VE3 / prior rank
- r76a r69a - U-net, r76b r69a - MLP gate, r76c r69a - selective smear (static), r76d (explore) hashed
  bigram memory (16384x128 -> 512) injected at layer 4 via a state-read gate, on r25a

Results R76: a no U-net 0.926608 @ 261.2 (U-net worth ~0 in r69a, saves no time); b no MLP gate
0.927166 @ 257.9 (gate worth 6e-4 for 3.2 s); c static smear 0.927129 @ 263.8 (selective worth 5.5e-4);
d GATED MID-NETWORK HASHED BIGRAM MEMORY (layer 4) on r25a 0.956230 (-0.0010 vs control, +1.4 s).

## Round 77 (92 left)
- r77a r69a + context-gated priors (per-token gate per prior order, from final state)
- r77b r69a - U-net + gated hashed bigram memory at layer 4
- r77c (explore) doc-reset running mean of wte projected into x0, on r25a
- r77d (explore) memory with bigram + trigram hashed lookups (shared table), on r25a

Results R77: a gated priors 0.925797 @ 262.5 (-0.0008, over 0.85 s); b no U-net + mid memory 0.925304
@ 262.5 (-0.0013, over); c DOC-MEAN x0 (doc-reset running mean of wte, 512x512 proj) on r25a 0.954226
(-0.0030 vs control, +5.4 s) STRONG NEW EXPLORE; d memory bi+trigram 0.955901 (-0.0003 vs r76d).

## Round 78 (88 left)
- r78a r77b + gated priors - MLP gate; r78b r77b + low-rank (128) doc-mean - MLP gate
- r78c (explore refine) low-rank doc-mean on r25a; r78d (explore) selective-time RoPE (per-head clock
  p = cumsum dt_h(x0), dt = 2 sigmoid(w.x0)) on r25a

Results R78: a r77b + gated priors - MLP gate 0.925327 @ 259.4, 52.7M = NEW BEST VALID (params now AT CAP
52.8M!); b + low-rank doc-mean in x0 0.928656 @ 262.5, 52.9M (worse, over both caps: doc-mean in x0 HURTS
the stack); c low-rank doc-mean on r25a 0.954559 @ 235.1 (-0.0027); d selective-time RoPE 0.956727 @ 238.5
(-0.0005, +8 s: weak).

## Round 79 (84 left)
- r79a r78a DROP_VE=3; r79b r78a + trigram lookup in memory
- r79c (explore) hidden-state doc-mean injected at layer 4 (rank 128) on r25a
- r79d (explore) document-unigram cache prior (rank 64) folded into head, on r25a

Results R79: a r78a DROP_VE=3 0.924342 @ 258.8, 52.7M = NEW BEST VALID; b + trigram in memory 0.925471 (no
gain); c hidden-state doc-mean at layer 4 on r25a 0.957992 (worse than control); d doc-unigram cache prior
on r25a 0.957613 (no gain). => document summaries help only as an x0 feature in the weak control.

## Round 80 (80 left)
- r80a r79a + MLP gate in layers 4-7; r80b r79a mem 8192 rows + bigram rank 256; r80c r79a mem at layer 2
- r80d (explore) selective token shift on MLP input, on r25a

Results R80: a r79a + MLP gate in layers 4-7 0.923610 @ 261.6 = NEW BEST VALID (0.06 s under cap!);
b mem 8192 + bigram rank 256 0.924619 (worse); c mem at layer 2 0.924663 (worse; layer 4 is right);
d SELECTIVE MLP TOKEN SHIFT on r25a 0.954930 (-0.0023 vs control, +5.5 s).

## Round 81 (76 left)
- r81a r80a - token meta (pricing), r81c r80a - stale depth gates (pricing)
- r81b r79a + selective MLP token shift in layers 4-7 (vs r80a's MLP gate there)
- r81d (explore) key salience bias via last q/k channel on r25a

Results R81: a r80a - token meta 0.924150 @ 260.6 (meta worth 5e-4 for 1.0 s); b r79a + selective MLP
token shift in 4-7 0.924397 (no gain in stack: smears already cover it); c r80a - stale depth gates
0.925173 @ 257.6 (gates worth 1.6e-3 for 4 s); d key salience (last q/k channel) on r25a 0.958113 (worse,
+10 s: negative).

## Round 82 (72 left): grouped versions of the cheap gates in r80a
- r82a channel-grouped (4) depth gates; r82b 12-group prior gates; r82c 8-group memory gate
- r82d (explore) per-token adaptive logit soft-cap on r25a

Results R82: a 4-group depth gates 0.924431 @ 261.8 (worse, over); b 12-group prior gates 0.923532 @ 261.6;
c 8-group MEMORY GATE 0.923481 @ 261.6 = BEST VALID (b, c within noise of r80a 0.923610); d adaptive
softcap on r25a 0.957937 (worse).

## Round 83 (68 left)
- r83a r82c + grouped prior gates; r83b r82c - ctx-x0 conv (pricing); r83c r82c - VE lag gates (pricing)
- r83d (explore) output embedding W_out[x_t] added to the input (per-channel lambda), on r25a

Results R83: a r82c + grouped prior gates 0.923469 @ 261.6 (52,761,640 params; cap ~52.816M) = BEST VALID
(marginal); b - ctx-x0 conv 0.923746 @ 260.1 (ctx-x0 now worth only 2.7e-4 for 1.5 s); c - VE lag gates
0.924627 @ 256.9 (worth 1.1e-3 for 4.7 s); d output emb W_out[x_t] as input on r25a 0.957199 (~0).

## Round 84 (64 left): spend the 1.5 s freed by dropping ctx-x0
- r84a + 2nd memory read at layer 6 (skip-bigram hash, shared table/proj); r84b + MLP gates layers 2-7;
  r84c + 3-lag selective ctx-VE; r84d (explore) output-side token metadata embedding on r25a

Results R84: a + 2nd memory read (skip-bigram) 0.923772 (no gain); b - ctx-x0 + MLP gates layers 2-7
0.923372 @ 260.6 = BEST VALID (marginal); c - ctx-x0 + 3-lag ctx-VE 0.923599 @ 260.1; d output-side token
metadata on r25a 0.958153 (worse). NOTE: we are in the noise regime (+-2e-4) with stack tweaks.

## Round 85 (60 left)
- r85a r84b + memory read reused as trigram-context logit prior (rank 64; skip rank 96->64)
- r85b r84b + MLP gates all 8 layers; r85c r84b + 3-lag ctx-VE
- r85d (explore) document occurrence count of x_t (sort-based, bucketed) embedding on r25a

Results R85: a memory read as trigram logit prior 0.923853 @ 262.8 (worse, over); b MLP gates all 8 0.923484;
c r84b + 3-lag VE 0.923753 (all noise/worse); d DOC OCCURRENCE COUNT of x_t (sort-based, bucketed
embedding) on r25a 0.955723 (-0.0015 vs control, +1.5 s) = promising explore.

## Round 86 (56 left)
- r86a r83a - ctx-x0 + occurrence count; r86b r84b with token meta replaced by occurrence count
- r86c (explore) bigram occurrence count on r25a; r86d (explore) count + log2 distance to prev occurrence

Results R86: a r83a - ctx-x0 + occ count 0.923154 @ 262.1 (over 0.45 s; -2e-4 vs r84b); b r84b meta->occ
0.923266 @ 262.1 (over); c BIGRAM occurrence count on r25a 0.955123 (-0.0021); d count + log2 distance
0.955025 (-0.0022). The sort costs ~2 s in the stack.

## Round 87 (52 left)
- r87a r83a - ctx-x0 + bigram occ count (16-bit hashed key, int32 sort); r87b r86a with int32 sort
- r87c (explore) unigram count + distance + bigram count on r25a
- r87d (explore) hard induction pointer: layer-2 state at successor of previous occurrence -> layer 4

Results R87: a + bigram occ (int32) 0.923062 @ 262.5 (over); b r86a int32 sort 0.922928 @ 262.0 (BEST
QUALITY, 0.34 s over cap); c all 3 occurrence features on r25a 0.954388 (-0.0029); d hard induction pointer
(successor-of-prev-occurrence state) 0.959621 (worse: negative, like r75).

## Round 88 (48 left)
- r88a r87b with MLP gates 5-7 (fit cap); r88b r88a + log2-distance embedding (same sort);
  r88c r87b + distance - token meta; r88d (explore) local repetition flags [x_t == x_{t-k}], k=1..4

Results R88: a r87b gates 5-7 0.923473 @ 266.4 (SLOWER: compile variance from irregular layers!); b + distance
0.923224 @ 266.9; c r87b + log2-distance - token meta 0.923063 @ 261.3 = BEST VALID; d local repetition
flags on r25a 0.956350 (-0.0009).

## Round 89 (44 left)
- r89a r88c + local repeat flags; r89b r88c with row-level (cross-doc) counts; r89c r88c + n-gram
  repetition depth (batched sort n=1..4); r89d (explore) count + distance + depth on r25a

Results R89: a + local repeat flags 0.923040 @ 263.7 (over); b row-level (cross-doc) counts 0.923493 (worse:
doc-level is right); c + N-GRAM REPETITION DEPTH (batched sort n=1..4) 0.922832 @ 263.0 (BEST QUALITY, 1.3 s
over); d depth on r25a 0.954575 (≈ r87c: no extra on control).

## Round 90 (40 left): fit r89c into the cap
- r90a r89c ranks 160+64; r90b r88c + bigram-only depth; r90c r89c MLP gates 4-6
- r90d (explore) doc-length-aware attention temperature on r25a

Results R90: a r89c ranks 160+64 0.923683 @ 260.7 (prior rank is worth a lot: +8.5e-4); b r88c + bigram-only
depth 0.923036 @ 262.1 (over 0.44 s); c r89c MLP gates 4-6 0.923149 @ 267.0 (SLOW: gate-matrix counts of 3
are slow (Muon sharding?), 4 or 6 are fine); d DOC-LENGTH ATTENTION TEMPERATURE on r25a 0.954848 (-0.0024,
zero time cost!).

## Round 91 (36 left)
- r91a r88c + doclen temp; r91b r90b + doclen temp; r91c r89c + doclen temp, ranks 176+80
- r91d (explore ablation) row-position temperature on r25a

Results R91: a r88c + doclen temp 0.922870 @ 262.9 (over); b r90b + doclen temp 0.922550 @ 263.0 (BEST QUALITY,
over 1.3 s); c r89c + doclen, ranks 176+80 0.923249 @ 261.6 (valid, worse than r88c); d ROW-position temp on
r25a 0.954847 = doc-length version (0.954848): the gain is position-dependent temperature (SSMax-like), not
document-relative. The doc version costs ~1.6 s in the stack (per-layer (B,T,H) tensors).

## Round 92 (32 left)
- r92a/b/c = r90b / r88c / r89c + static row-position temperature (q *= 1 + a_h log(1+t)/log T; (T,) table)
- r92d (explore) multiplicative per-key salience k *= 2 sigmoid(w_h . x[:32]) on r25a

Results R92: static row temp (multiply on q after the norm) is SLOW in the stack: a r90b+ 0.922549 @ 265.5,
b r88c+ 0.922931 @ 264.2, c r89c+ 0.922637 @ 264.6 (all over); d MULTIPLICATIVE KEY SALIENCE
k *= 2 sigmoid(w.x[:32]) after QK-norm on r25a 0.955189 (-0.0021, ~free).

## Round 93 (28 left)
- r93a r90b / r93b r88c + temperature folded into the query RoPE tables (norm first, then rotate with
  cos*s, sin*s: same math, verified numerically); r93c r88c + key salience (extra smear-gate columns)
- r93d (explore) per-key value salience on r25a

Results R93: a r90b + RoPE-folded temp 0.922696 @ 264.5, b r88c + RoPE-folded temp 0.922918 @ 262.5 (still
over: the temperature costs ~1-3 s in the stack whatever the form); c r88c + KEY SALIENCE (4 extra columns of
the smear-gate matmul) 0.923007 @ 249.4 = BEST VALID and 12 s FASTER (timing artifact of the gate matrix
shape 8x32 -> 12x32 in Muon, not engineered); d value salience on r25a 0.955923 (-0.0013).

## Round 94 (24 left): spend r93c's headroom on the features that were over the cap
- r94a r93c + RoPE-folded temperature + n-gram depth4; r94b r94a + token meta + ctx-x0;
  r94c r94a + token meta + MLP gates all 8; r94d (explore) key salience + row temp on r25a

Results R94: a r93c + RoPE-folded temp + n-gram depth4 0.922335 @ 251.4; b + token meta + ctx-x0 0.921630 @ 255.1
(52,774,344 params; cap 52.849M) = BEST VALID; c a + meta + MLP gates all 8 0.921942 @ 253.4; d key salience +
row temp on r25a 0.955129 (NOT additive on the control: ~ either alone).

## Round 95 (20 left)
- r95a r94b + MLP gates all 8; r95b r95a + bigram occurrence count (free from the batched sort);
  r95c r95a + 3-lag ctx-VE; r95d (explore) key + value salience on r25a

Results R95: a r94b + MLP gates all 0.921678 @ 254.4 (~0); b + bigram occ 0.921747 (~0); c a + 3-lag ctx-VE
0.921402 @ 255.2 = BEST VALID; d KEY + VALUE SALIENCE on r25a 0.952950 (-0.0043 vs control: additive, and fast).

## Round 96 (16 left)
- r96a r95c + value salience (smear-gate matmul 16 cols); r96b r96a + n-gram depth up to 6;
  r96c r96a + per-query temperature (5th block); r96d (explore) LEXICAL salience table vs contextual on r25a

Results R96: a r95c + value salience 0.921542 @ 259.7 (no gain in stack, slower); b + depth6 0.921745; c + per-query
temperature 0.921711 (no gain); d LEXICAL salience table on r25a 0.955808 (vs contextual 0.952950: salience is
contextual, not a token property).

## Round 97 (12 left): revive pointer-sentinel + full-layer doc mask in the r95c headroom (6.4 s)
- r97a r95c + pointer + doc mask (full layers); r97b pointer only; r97c doc mask only
- r97d (explore) key/value salience only in the full layers on r25a

Results R97: pointer now HURTS in the stack: a pointer + doc mask 0.922204 @ 265.6, b pointer 0.922490 @ 262.0 (both
over); c doc mask only 0.921499 (~0); d salience only in the 2 full layers on r25a 0.956060 (-0.0012 vs -0.0043 in
all layers: salience matters in local layers too).

## Round 98 (8 left)
- r98a r95c 4-lag VE; r98b r95c selective (gated) ctx-x0 conv lags; r98c r95c temperature ablated (x0 slope)
- r98d (explore) signed value salience 1 + 2 tanh(.) in (-1, 3) on r25a

Results R98: a 4-lag VE 0.921446 (~0); b selective ctx-x0 lags 0.921592 @ 279.5 (slow, no gain); c r95c with the
position temperature ablated 0.921689 (temperature worth ~2.9e-4); d signed value salience on r25a 0.952817
(~ unsigned 0.952950).

## Round 99 (4 left after this)
- r99a r95c replicate (noise); r99b r95c + U-net; r99c r95c + (depth, count) embedding at the head input
- r99d (explore) control + all control-level wins (occ all3 + key/value salience + pos temp + MLP shift)

Results R99: a r95c REPLICATE 0.921488 (vs 0.921402: noise ~1e-4); b + U-net 0.921448 (~0); c (depth,count) at head
0.921564 (~0); d control + ALL control wins 0.949723 (-0.0075 vs control; components sum to -0.012: ~60% additive).

## Round 100 (final 4)
- r100a r95c + key-position temperature (key RoPE tables); r100b r95c + depth-conditioned prior gates;
  r100c both; r100d (explore) key-position temperature on r25a (+ query temp)

Results R100 (final): a r95c + KEY-POSITION TEMPERATURE 0.921311 @ 258.2 = BEST VALID (within noise); b depth-conditioned
prior gates 0.921548 (~0); c both 0.921495; d key-position temp on r25a (+ query temp) 0.955080 (~ query temp alone).

## Ideas backlog
- per-head heterogeneous windows in S layers; smeared keys (prev-token key mix); attention output gate;
  U-net skips; value norm; lm_head unigram bias; forget-gate/decay bias.
