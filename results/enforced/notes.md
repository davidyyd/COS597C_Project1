# Notes

## SUMMARY (final, budget 400 used)
FINAL BEST: r94a-best-blockdiag 0.92868 (repeats 0.92905, 0.92882 -> mean 0.92885); identical function: r87a 0.92888
(repeat 0.92878). Baseline 0.95895 -> -0.0301 bpb (about -3.1%). The model is at the 68.9 ms step cap (67.9 ms).
What the final model is (all on the baseline GPT, 8x512, same params/FLOPs budget):
  A. gated normalized value path (VE layers 1,3,5,7: v = s_h(g1 norm(W_v u) + g2 norm(VE[t])), g = 2 sigmoid(.)),
     the single biggest win (-0.011 on baseline): sparse-update embedding tables must be normalized at use.
  B. gates as gates: all small gate matrices trained by AdamW ("other"), not Muon.
  C. head self-gate y_h *= 2 sigmoid(a_h.y_h + U_h V_h y_h + B[token,l,h]) (low-rank in 1,3,5,7, computed as
     block-diagonal matmuls), per-token head-gate bias table B.
  D. undiluted (normalized) reads of recent sublayer outputs, incl. the previous position (keys read m_{l-1}[t-1]
     except in the isolated layers; MLP reads a_l[t], a_l[t-1], m_{l-1}[t]).
  E. deep embedding, x0 re-injection gate, MLP input self-gates in layers 4-7 (+ token gate features 5-7).
  F. document awareness: doc-position query temperature, VE gate and output temperature terms, and
     DOCUMENT MASKING IN THE EARLY LOCAL LAYERS ONLY (0-2).
  G. multi-scale head windows 128..1024 in local layers with RoPE base proportional to the window (20 x window,
     40000 in full layers).
Final-model ablations (R97, vs mean 0.92885): no early doc mask +0.0014; RoPE base 10000 everywhere +0.0008;
no token head-gate table +0.0003; no token MLP-gate features +0.0002. Transfer to a different (cheaper) model r92b (R100): RoPE ~ window -0.0012,
token head-gate table -0.0010 (both stack: -0.0017), early doc mask worth 0.0026; multi-scale windows HURT there (+0.0033).
KEY FINDINGS of the second half (R70-R101):
  1. DEPTH-SPLIT DOCUMENT ISOLATION: early local layers want only the current document (mask 0-2: -0.0028 on the
     cheap base), late layers want the neighbouring document (masking them hurts; restricting late/global layers to
     current+previous document = unrestricted). Isolation not locality (whole-document early attention = windowed).
     Isolated layers use their document's BOS as a sink (hiding it +0.0008). A learned soft bias (doc-parity q/k
     features) cannot replace the hard mask. No prior work found for the depth split.
  2. RoPE base proportional to each head's receptive field (multi-scale heads 128..1024): small but robust gain,
     transfers. Larger constant (x2) better than x1, x4 saturates.
  3. Per-token tables steering gates (head-gate biases, MLP gate features) are a cheap, FLOP-free use of the
     embedding cap: small gains, transfer.
  4. Per-head attention-output normalization with a learned per-head scale helps models WITHOUT the full gating
     stack (-0.0012 cheap base; the per-head scale is essential) but hurts the best model.
  5. Speed: block-diagonal dense matmuls for per-head low-rank maps are ~0.5 ms faster than einsums; score_mod
     doc masks cost ~0.2 ms/layer; create_block_mask in the forward fails the precision check (int8).
Cheap bases for explores: r83a-vag-dm-ingate 0.94006 (60 ms); r92b (+ head gate + y-norm) 0.93737 (67 ms).
Earlier state (r61a era): r61a-lrodd-outgate-late 0.93223 (repeat 0.93208) = r59a with low-rank gate in odd layers + per-channel MLP write gates
(m *= 2 sigmoid(U V u), rank 32, AdamW, from the MLP input) in layers 4-7. Base recipe (r59a 0.93278) = baseline +
  1. GATED NORMALIZED VALUE PATH (biggest win; RMS-normalizing the value embeddings alone gives -0.011 on the
     baseline): VE layers v = s_h (g1 norm(W_v u) + g2 norm(VE[t])), g = 2 sigmoid(W x); other layers
     v = s_h norm(W_v u). Mechanism: VE rows are sparse (updated only when their token occurs) under AdamW weight
     decay; init-at-unit-scale without normalization recovers only -0.0014.
  2. head self-gate y_h *= 2 sigmoid(a_h . y_h + U_h V_h y_h) (rank 32 per head) in all layers
  3. gate matrices (VE/proj gates, self-gate vectors and low-rank maps) labelled "other" -> AdamW: better AND
     faster than Muon (Muon spends time on many tiny matrices)
  4. undiluted, time-shifted reads: attention q,v read norm(x) + b1 norm(m_{l-1})[t]; keys additionally read
     b2 norm(m_{l-1})[t-1]; MLP reads norm(x) + b3 norm(a_l)[t] + b4 norm(a_l)[t-1] + b5 norm(m_{l-1})[t]
  5. deep embedding x0 = norm(e + MLP_512(e)); x0 re-injection gate (1 + tanh(u . norm x))
  6. doc-position query temperature (f_doc = log(1+t-doc_start)/log T); per-token output temperature before
     the 15 softcap (softcap essential)
  7. multi-scale heads in the sliding layers (windows 128/256/512/1024)
Cheap bases: r58b (baseline + value path, AdamW gates: 0.94618, 58 ms); r50c-like compositions (63 ms).
Principles: (P1) gate attention outputs from what was retrieved; (P2) bounded temperatures on the query/output
side; (P3) undiluted reads of recent sublayer outputs, incl. the previous position (keys benefit, queries not);
(P4) token-identity shortcuts (x0 gate, deep embed); (P5) document position; (P6) normalize every value source,
mix with token gates; (P7) gates are gates: train them with AdamW.
Consistently bad: per-token multipliers on MLP/residual writes from unbounded or foreign signals (but a per-channel
gate from the MLP's own normalized input helps a bit), competition across heads, key/value per-token scales,
mid-depth renorm, coupled head, signed square, removing the softcap, VE placement other than 1,3,5,7, VE in
all layers.
Time: cap 68.9 ms, noise +-1.5 ms; best model at the cap. Self-value skip costs ~3 ms (!).

## Practical
- ar-exp: run with Bash run_in_background (timeout 3600000) then wait with an until-loop on result.json.
- Patch helper: /tmp/patch.py (patch(src, dst, [(old,new),...])) — recreate if gone.
- Step time: per-layer elementwise ops on the [B,T,512] residual cost ~0.5 ms/layer (F.pad shift in every
  layer -> 70 ms, fail). Some models measure ~60-61 ms (selfgate, qtemp) — unexplained, maybe the
  benchmark node; baseline listed as 66.

## Results
| round | name | parent | val_loss | note |
|---|---|---|---|---|
| 1 | r01a-prevtok-key | base | 0.95778 | k input += lambda*x0_{t-1}, layers 0,2,4,6 only (speed) |
| 1 | r01b-attn-selfgate | base | **0.95653** | y_h *= 2 sigmoid(a_h . y_h), 61 ms |
| 1 | r01c-qtemp | base | diverged | q *= exp(W x) after QK-norm: unbounded -> NaN at step 458 |
| 1 | r01d-ve-keys | base | 0.95759 | k += tanh(gate)*ve in VE layers |
| 2 | r02a-selfgate-vekeys | r01b | 0.95675 | ve-keys does not add on selfgate |
| 2 | r02b-selfgate-prevtok | r01b | 0.95568 | prevtok all 8 layers, -0.0009 vs r01b, 65 ms |
| 2 | r02c-selfgate-qtanh | r01b | **0.95353** | q *= 1+tanh(W x) after QK-norm, -0.003 vs r01b |
| 2 | r02d-multiscale-heads | base | 0.95520 | S-layer head windows 128/256/512/1024, -0.0037 vs base, 63 ms |
| 3 | r03a-qtanh-multiscale | r02c | 0.95473 | multiscale does NOT stack with qtemp (+0.0012), 57 ms |
| 3 | r03b-qtanh-prevtok | r02c | **0.95282** | prevtok keys all layers, -0.0007, 63 ms |
| 3 | r03c-qtanh-ktemp | r02c | 0.95665 | per-key salience 1+tanh hurts (+0.003) |
| 3 | r03d-mlp-selfgate | base | 0.98948 | MLP out *= 2sig(c.out): very bad (unbounded y, gate saturates) |
| 4 | r04a-gate-xy | r03b | 0.95271 | gate += w.x term: ~no change |
| 4 | r04b-prevtok-qk | r03b | 0.95303 | shared shifted input to q and k: no gain |
| 4 | r04c-out-temp | base | 0.95644 | logits *= 1+tanh(w.norm x): -0.0025 vs base |
| 4 | r04d-x0gate | base | 0.95636 | x0 weight *= 1+tanh(u.norm x): -0.0026 vs base |
| 5 | r05a-best-outtemp | r03b | 0.95447 | out-temp HURTS on top of qtemp/selfgate (+0.0016) |
| 5 | r05b-best-x0gate | r03b | **0.95219** | x0gate adds -0.0006..-0.0009 |
| 5 | r05c-r03b-repeat | r03b | 0.95312 | identical code to r03b (0.95282): noise ~0.0003 |
| 5 | r05d-mlp-intemp | base | 0.96083 | MLP input *= 1+tanh(w.u): hurts (+0.002) |
| 6 | r06a-vstrength | r05b | 0.95718 | v *= 1+tanh(Wx): HURTS +0.005 |
| 6 | r06b-prevtok2 | r05b | 0.95246 | + x0_{t-2} in keys: no gain |
| 6 | r06c-ctxmean | base | 0.95760 | x += b_i norm(cumsum x0): -0.0014 vs base, 69 ms (at cap!) |
| 6 | r06d-ve-shared-all | base | 0.96029 | VE every layer, pair-shared tables: hurts +0.0013 |
| 7 | r07a-resid-gate | r05b | 1.04060 | per-token scale on carried residual: CATASTROPHIC |
| 7 | r07b-best-ctxmean | r05b | 0.95227 | ctxmean no gain on best |
| 7 | r07c-qtemp-logpos | r05b | 0.95246 | log-position term in q temp: no gain |
| 7 | r07d-centered-relu2 | base | 0.96009 | h - mean(h) before relu^2: hurts +0.001 |
| 8 | r08a-ms-noqtemp | r05b | 0.95284 | swap qtemp for multiscale: slightly worse |
| 8 | r08b-lowrank-selfgate | r05b | **0.95027** | + per-dim gate U_h V_h y_h (rank 32 per head), 68 ms |
| 8 | r08c-token-qtemp | r05b | 0.95212 | token-identity offset to qtemp: no gain |
| 8 | r08d-head-compete | base | 0.98396 | softmax over heads: CATASTROPHIC |
| 9 | r09a-gate-multiscale | r08b | **0.95006** | + multiscale windows: ~same/slightly better, 66 ms (faster) |
| 9 | r09b-gate-rank48 | r08b | 0.95026 | rank 48: no change |
| 9 | r09c-crosshead-gate | r08b | 0.95307 | cross-head rank-48 gate (2-D Muon): WORSE +0.003 |
| 9 | r09d-mlp-x0 | base | 0.95679 | MLP input = norm(x) + beta x0: -0.0022 vs base |
| 10 | r10a-ms-mlpx0 | r09a | 0.95095 | mlp-x0 on best: worse +0.0009 |
| 10 | r10b-qfeat-gate | r09a | **0.94946** | q *= 2sig(U_h V x) per-dim query gate, rank 16, 68.6 ms (AT CAP) |
| 10 | r10c-mlp-reads-attn | base | 0.95379 | MLP input = norm(x+a) + beta norm(a): -0.0052 vs base (big) |
| 10 | r10d-lowrank-gate-base | base | 0.95412 | scalar+lowrank self-gate alone: -0.0048 vs base |
| 11 | r11a-ms-mra | r09a | **0.94787** | + MLP reads attn (mra): -0.0022, 67 ms |
| 11 | r11b-qfeat-mra | r10b | 0.94878 | r10b - x0gate + mra (x0gate > qfeat given mra), 69 ms |
| 11 | r11c-mra-multiscale | r10c | 0.95283 | base + mra + multiscale: -0.001 vs r10c, 64 ms |
| 11 | r11d-head-reads-mlp | base | 0.95751 | head reads norm(x)+beta norm(m_L): -0.0014 vs base |
| 12 | r12a-best-headmlp | r11a | 0.95098 | head reads last MLP on best: HURTS +0.003 |
| 12 | r12b-best-attn-reads-mlp | r11a | **0.94688** | + attention reads norm(m_{l-1}): -0.001, 68 ms |
| 12 | r12c-mra-perchannel | r11a | 0.94751 | per-channel beta: -0.0004 (marginal) |
| 12 | r12d-no-embed-norm | base | 0.95721 | x0 = wte[idx] without norm: -0.0017 vs base |
| 13 | r13a-chain-perchannel | r12b | 0.94599 | per-channel read weights: -0.0009 |
| 13 | r13b-chain-noembnorm | r12b | 0.94693 | no embed norm on best: no effect |
| 13 | r13c-mlp-reads-prevmlp | r12b | **0.94522** | MLP also reads norm(m_{l-1}): -0.0017, 68.6 ms (cap) |
| 13 | r13d-ve-prevtok | base | 0.95805 | layer-1 VE indexed by prev token: hurts |
| 14 | r14a-noprev-perchannel | r14b | 0.94535 | per-channel on noprev: -0.0004 |
| 14 | r14b-chain3-noprev | r13c | 0.94579 | remove prev-token keys: +0.0006 (67 ms) |
| 14 | r14c-chain3-attn-reads-attn | r13c | **0.94455** | -prevtok + attention reads norm(a_{l-1}), 68.3 ms |
| 14 | r14d-attn-scales-mlp | base | 0.96525 | MLP write *= 1+tanh(w.norm a): BAD |
| 15 | r15a-chain4-nox0gate | r14c | 0.94619 | drop x0 gate: +0.0016 (x0 gate still matters) |
| 15 | r15b-chain4-Lmultiscale | r14c | 0.94657 | multiscale in full layers: +0.002 (keep global heads) |
| 15 | r15c-chain4-rawreads | r14c | diverged | reads without norm: diverged |
| 15 | r15d-window-rope | r02d | 0.95474 | RoPE base ~ head window: -0.0005 (noise) |
| 17 | r17a-r11a-attn-reads-attn | r11a | 0.94682 | r11a + attn<-a_{l-1}: -0.001 |
| 17 | r17b-query-reads-x0 | base | 0.95700 | q = W_q(u + beta x0): -0.002 vs base |
| 17 | r17c-best-only-last-full | r14c | 0.94643 | layer 3 sliding multiscale: +0.0019 (keep 2 full layers) |
| 17 | r17d-mid-renorm | base | 0.96740 | x=norm(x) at mid-depth: BAD +0.0085 |
| 18 | r18a-scalargate-prevtok | r14c | 0.94598 | drop lowrank gate, add prevtok: +0.0014 but 65.4 ms (cheap base) |
| 18 | r18b-scalargate-prevtok-qfeat | r18a | 0.94494 | + qfeat rank16: -0.001, 68.6 ms |
| 18 | r18c-scalargate-prevtok-qx0 | r18a | 0.94612 | + query reads x0: no gain |
| 18 | r18d-doc-position | base | 0.95686 | x0 += log(1+t-docstart)/log T * w: -0.0021 vs base |
| 19 | r19a-qfeat-docpos | r18b | 0.94468 | + docpos input: -0.0003 (68.9 ms) |
| 19 | r19b-scalargate-docpos | r18a | 0.94576 | + docpos input: -0.0002 |
| 19 | r19c-docpos-scalarprev | r19b | 0.94614 | scalar prev lambda: +0.0004, no time saved |
| 19 | r19d-docpos-temp | base | 0.95467 | q *= 1+tanh(c_h f_doc): -0.0043 vs base (strong explore) |
| 20 | r20a-doctemp-mra | r19d | 0.95178 | doctemp + mra: cheap lineage, 66 ms |
| 20 | r20b-best-doctemp | r14c | **0.94439** | doc term in qtemp: -0.00016 (noise), 68.5 ms |
| 20 | r20c-scalargate-doctemp | r18a | 0.94609 | no gain |
| 20 | r20d-doc-outtemp | base | 0.95698 | logits *= 1+tanh(a+b f_doc): -0.002 vs base |
| 21 | r21a-cheap-gate-emachain | r21b | 0.94873 | decaying-sum read: worse than pairwise chain |
| 21 | r21b-cheap-lineage-gate | r20a | 0.95187 | + scalar gate: no gain (62.5 ms!) |
| 21 | r21c-cheap-gate-chain | r21b | 0.94791 | + full pairwise chain: -0.004, 65.4 ms (cheap lineage) |
| 21 | r21d-doc-x0 | base | 0.95714 | x0 lambda *(1+tanh(c f_doc)): -0.0018 vs base |
| 22 | r22a-cheap-tempscale | r21c | 0.94787 | per-head temp scale: no gain |
| 22 | r22b-cheap-ms-x0gate | r21c | 0.94672 | + multiscale + x0gate: -0.0012, 64.5 ms |
| 22 | r22c-cheap-prevtok | r21c | 0.94771 | prevtok: -0.0002 |
| 22 | r22d-bigram-logits | base | 0.95926 | low-rank bigram logit path: no gain |
| 23 | r23a-cheap-qtemp | r22b | 0.94639 | + content qtemp: -0.0003, 64 ms |
| 23 | r23b-cheap-lowrank | r22b | 0.94494 | + lowrank gate: -0.0018, 68.8 ms |
| 23 | r23c-cheap-qtemp-prevtok | r23a | 0.94591 | + prevtok: -0.0005, 65.7 ms |
| 23 | r23d-deep-embed | base | 0.95503 | x0 = norm(e + MLP_512(e)): -0.0039 vs base (strong explore) |
| 24 | r24a-qtp-deepemb | r23c | 0.94560 | deep-embed: -0.0003, 67.3 ms |
| 24 | r24b-cheap-deepemb | r22b | 0.94547 | deep-embed: -0.0012, 66.1 ms |
| 24 | r24c-doctemp-mra-deepemb | r20a | 0.95004 | deep-embed: -0.0017 |
| 24 | r24d-bigram-deepemb | base | 0.95442 | MLP over [e_t; e_{t-1}] (hidden 256): -0.0006 vs unigram deep-embed |
| 25 | r25a-cheap-de-lowrank-half | r24b | 0.94441 | lowrank gate in odd layers only: -0.0011, 68.1 ms (ties best) |
| 25 | r25b-cheap-de-qtemp | r24b | 0.94607 | content qtemp: no gain |
| 25 | r25c-cheap-bigramde | r24b | 0.94671 | bigram deep-embed: no gain on chain model |
| 25 | r25d-deep-head | base | 0.95844 | token MLP before head: -0.0005 (no) |
| 26 | r26a-vegate-full | r25a | **0.94391** | VE gate reads all 512 channels (not 32): -0.0005 |
| 26 | r26b-coupled-head | base | 0.96334 | W_out = W_head + E A B: BAD +0.0044 |
| 26 | r26c-best-halflr-deepemb | r20b | 0.94497 | half lowrank + deep-embed on r20b: worse |
| 26 | r26d-no-attn0 | r24d | 0.95773 | drop layer-0 attention: +0.0033 (needed) |
| 27 | r27a-r20b-vegate | r20b | 0.94408 | VE full gate on r20b: -0.0003 |
| 27 | r27b-best-vegate-qtemp | r26a | 0.94440 | content qtemp on r26a: no |
| 27 | r27c-qgated-partial-rope | base | 0.95952 | query-gated partial RoPE: +0.0006 (no) |
| 27 | r27d-signed-square | base | 0.97618 | h|h| activation: very bad |
| 28 | r28a-best-outtemp | r26a | **0.94335** | content out-temp: -0.0006 (works w/o content qtemp) |
| 28 | r28b-best-docouttemp | r26a | 0.94352 | doc out-temp: -0.0004 |
| 28 | r28c-best-vekeys | r26a | 0.94419 | VE into keys: no |
| 28 | r28d-mlp-reads-ve | base | 0.95940 | MLP reads VE: no |
| 29 | r29a-outtemp-doc | r28a | 0.94336 | + doc term in out-temp: no |
| 29 | r29b-r28a-repeat | r28a | 0.94354 | REPEAT of r28a: noise 0.0002 -> r28a real (~0.9434) |
| 29 | r29c-values-read-x0 | base | 0.95853 | values read x0 (non-VE layers): -0.0004 (no) |
| 29 | r29d-stretch-rope | r01b | 0.95637 | content-dependent RoPE step: no (-0.0002 vs r01b) |
| 30 | r30a-best-no-deepemb | r28a | 0.94540 | remove deep-embed: +0.002 (important here) |
| 30 | r30b-best-qfeat-swap | r28a | 0.94452 | half-lowrank -> qfeat: +0.0012 |
| 30 | r30c-best-prevtok-swap | r28a | 0.94490 | deep-embed -> prevtok: +0.0015 |
| 30 | r30d-mult-read | r10c | 0.95523 | multiplicative MLP read: worse than additive |
| 31 | r31a-best-bigramde | r28a | 0.94411 | bigram deep-embed on best: worse |
| 31 | r31b-best-de-docpos | r28a | 0.94413 | deep-embed sees doc pos: worse |
| 31 | r31c-novelty-temp | r01b | 0.95429 | q *= 1+tanh(c nov), nov=1-cos(x0, mean prev x0): -0.0022 vs r01b |
| 31 | r31d-ema-context | r01b | 0.95685 | multi-timescale EMA context: no |
| 32 | r32a-novtemp-swap | r28a | 0.94485 | half-lowrank -> novelty temp: +0.0015 (lowrank better) |
| 32 | r32b-novout-swap | r28a | 0.94585 | half-lowrank -> novelty out-temp: worse |
| 32 | r32c-nov-x0 | r01b | 0.95613 | novelty-modulated x0: -0.0004 |
| 32 | r32d-repeat-temp | r01b | 0.95571 | exact repeat-count temperature: -0.0008 (novelty better) |
| 33 | r33a-nocap-vegate-doc | r33b | 0.94953 | (no cap) |
| 33 | r33b-best-nocap | r28a | 0.94952 | remove logit softcap: +0.006 -> SOFTCAP ESSENTIAL |
| 33 | r33c-head-topic | base | 0.95737 | head reads causal topic sum: +0.0016 (bad) |
| 33 | r33d-self-value-skip | r01b | 0.95541 | y_h += g_h v_h(t): -0.0011 vs r01b |
| 34 | r34a-postcap-temp | r28a | 0.94426 | out-temp after cap: worse |
| 34 | r34b-both-temps | r28a | 0.94510 | temps before and after cap: worse |
| 34 | r34c-head-x-x0 | base | 0.95751 | head reads norm(x*x0): no |
| 34 | r34d-value-norm | r01b | 0.95497 | v <- s_h norm(v): -0.0016 vs r01b |
| 35 | r35a-lr37-vnorm | r28a | **0.94107** | lowrank only layers 3,7 + VALUE NORM: -0.0023 (new best) |
| 35 | r35b-nolr-vnorm | r28a | 0.94174 | no lowrank + value norm: -0.0016 |
| 35 | r35c-silu-values | r01b | 0.95546 | v *= 2sig(s v): -0.0011 vs r01b |
| 35 | r35d-keys-read-x0 | r01b | 0.95631 | keys read x0: no |
| 36 | r36a-vnorm-pre-ve | r35a | 0.94043 | norm proj before adding raw VE: -0.0006 |
| 36 | r36b-nolr-ve-sep | r35b | **0.93916** | v = s norm(norm(proj) + g norm(VE)): -0.0026 vs r35b, new best |
| 36 | r36c-nolr-novtemp | r35b | 0.94192 | novelty temp: no |
| 36 | r36d-ynorm-gate | r01b | 0.95427 | head writes norm(y)*gate: -0.0023 vs r01b |
| 37 | r37a-vesep-lr37-noot | r36b | 0.94058 | -outtemp +lowrank(3,7): worse |
| 37 | r37b-vesep-ynorm-noot | r36b | 0.93987 | -outtemp +norm(y)*gate: +0.0007 |
| 37 | r37c-vesep-noouter | r36b | **0.93796** | v = s(norm(proj) + g norm(VE)) no outer norm: -0.0012 |
| 37 | r37d-grouped-norm | r01b | 0.95671 | grouped RMSNorm inputs: no |
| 38 | r38a-gated-proj | r37c | **0.93717** | v = s(g1 norm(proj) + g2 norm(VE)): -0.0008 |
| 38 | r38b-ve-keys-norm | r37c | 0.94036 | normalized VE into keys: worse |
| 38 | r38c-signed-vegate | r37c | 0.93863 | signed VE gate (0 init): worse |
| 38 | r38d-norm-value-residual | r01b | 0.95556 | normalized value residual: -0.001 |
| 39 | r39a-ve-all-pairs-noot | r38a | 0.93991 | VE all layers pair-shared (no outtemp): worse |
| 39 | r39b-proj-gate-all | r38a | 0.93758 | value loudness gate in non-VE layers: no |
| 39 | r39c-prev-ve-keys-noot | r38a | 0.93764 | prev-token VE in keys (no outtemp): ~neutral |
| 39 | r39d-base-gated-values | r01b | 0.94631 | r38a VALUE PATH on r01b: -0.0102 (!!) 63 ms |
| 40 | r40a-nodeep-lr37 | r38a | 0.93785 | deep-embed -> lowrank(3,7): +0.0007 |
| 40 | r40b-gv-chain | r39d | 0.94330 | + chain reads: -0.003, 66.4 ms |
| 40 | r40c-gv-temps | r39d | 0.94361 | + doc temp + out temp: -0.0027, 63 ms |
| 40 | r40d-gv-ms-x0gate | r39d | 0.94332 | + multiscale + x0gate: -0.003, 62 ms |
| 41 | r41a-gv-chain-tokgated | r40b | 0.94506 | token-gated reads: +0.0018 (worse) |
| 41 | r41b-best-tokgated-reads | r38a | **0.93659** | token-gated reads: -0.0006 (contradicts r41a -> ~noise) |
| 41 | r41c-best-gates-from-x0 | r38a | 0.93859 | value gates from x0: worse |
| 41 | r41d-gv-prev-ve-values | r39d | 0.94584 | prev-token VE as 3rd value source: -0.0005 |
| 42 | r42a-gv-mlp-reads-ve | r39d | 0.94564 | MLP reads normalized VE: -0.0007 |
| 42 | r42b-gv-q-ve | r39d | 0.94658 | queries get normalized VE: no |
| 42 | r42c-gv-ve-all-pairs | r39d | 0.94915 | VE all layers pair-shared: +0.003 worse |
| 42 | r42d-gv-ve-half-all | r39d | 0.94750 | half-width VE in all layers: +0.0012 worse |
| 43 | r43a-base-valuepath | base | 0.94690 | value path on plain baseline: -0.0121 (!) 61 ms |
| 43 | r43b-base-ve-norm-only | base | 0.94784 | ONLY v = W_v u + g norm(VE): -0.0111 -> it's the VE SCALE |
| 43 | r43c-base-proj-norm-only | base | 0.95555 | only norm values after mix: -0.0034 |
| 43 | r43d-gv-small-ve-table | r39d | 0.94571 | small shared 128-d VE table for non-VE layers: -0.0006 |
| 44 | r44a-best-no-outtemp | r41b | 0.93902 | remove out-temp: +0.002 (important) |
| 44 | r44b-best-no-chain | r41b | 0.93926 | remove read chain: +0.0023, saves 5 ms (63.6 ms) |
| 44 | r44c-token-mlp-bias | r43a | 0.94479 | MLP hidden += b_l U norm(E[idx]) (8192x128 table): -0.0021, but +7.7 ms |
| 44 | r44d-r41b-repeat | r41b | 0.93717 | repeat of r41b (0.93659): noise ~0.0006 -> r41b ~ r38a ~ 0.9369 |
| 45 | r45a-nochain-mra | r44b | 0.93819 | chain -> only MLP reads attn: -0.0011 vs no chain, 64.3 ms |
| 45 | r45b-nochain-tb03 | r44b | 0.93827 | no chain + token MLP bias layers 0-3: -0.001, 66.2 ms |
| 45 | r45c-va-tb03 | r43a | 0.94526 | token bias layers 0-3: -0.0016 |
| 45 | r45d-va-bigram-tb03 | r43a | 0.94481 | bigram-hashed token bias layers 0-3: -0.0021 |
| 46 | r46a-nochain-mra-tb03 | r45b | 0.93848 | mra + tb03 instead of chain: worse than chain |
| 46 | r46b-nochain-mra-bigramtb03 | r45b | 0.93778 | mra + bigram tb03: ~chain level (0.9369) - no |
| 46 | r46c-base-ve-init-unit | base | 0.95753 | VE init unit RMS, no norm: only -0.0014 -> NORMALIZATION AT USE is the fix |
| 46 | r46d-va-token-outtemp | r43a | 0.94560 | out-temp with token-table term: -0.0013 |
| 47 | r47a-best-de-to-bigramtb | r38a | 0.93776 | deep-embed -> bigram tb03: ~same |
| 47 | r47b-va-docpos-values | r43a | 0.94647 | doc-pos bucket table into values: no |
| 47 | r47c-va-docpos-keys | r43a | 0.94738 | doc-pos bucket table into keys: no |
| 47 | r47d-va-ve-even | r43a | 0.95312 | VE on layers 0,2,4,6: +0.0062 WORSE -> placement matters (late/last) |
| 48 | r48a-va-ve-4567 | r43a | 0.95620 | VE layers 4-7: +0.009 worse |
| 48 | r48b-va-ve-plus6 | r43a | 0.94772 | + layer 6 sharing 7's lookup: +0.0008 |
| 48 | r48c-best-ve-4567 | r38a | 0.94360 | best with VE 4-7: +0.0064 worse |
| 48 | r48d-va-ve-3567 | r43a | 0.95460 | VE 3,5,6,7: +0.0077 worse -> alternating 1,3,5,7 is right |
| 49 | r49a-va-qtemp | r43a | 0.94576 | content qtemp: -0.0011 (+1 ms) |
| 49 | r49b-va-deepemb | r43a | 0.94410 | deep embed: -0.0028 (+1.6 ms) - best value per ms |
| 49 | r49c-va-lowrank | r43a | 0.94471 | scalar+lowrank self-gate: -0.0022 (+6.5 ms) |
| 49 | r49d-va-algebraic-cap | r43a | 0.94688 | algebraic softcap: same as tanh |
| 50 | r50a-compA | r43a | 0.93807 | deep+temps+ms+x0gate+mra+bigram tb03: 63 ms |
| 50 | r50b-compA-qtemp | r50a | 0.93833 | + content qtemp: no |
| 50 | r50c-compA-fullchain | r50a | **0.93716** | full chain instead of mra+tb: = best level at 63.3 ms (5 ms cheaper) |
| 50 | r50d-va-bigram-values | r43a | 0.94606 | bigram-hashed value table (3rd source): -0.0008 |
| 51 | r51a-c-bigramtb | r50c | 0.93712 | + bigram tb03: no |
| 51 | r51b-c-gate-halflr | r50c | **0.93611** | + scalar gate + lowrank odd layers: -0.001, 67.8 ms (NEW BEST) |
| 51 | r51c-c-scalargate | r50c | 0.93761 | + scalar gate only: no |
| 51 | r51d-c-shift-read | r50c | 0.93647 | attn reads norm(m_{l-1}) at position t-1: -0.0007, 65 ms |
| 52 | r52a-gate-shift-swap | r51b | **0.93572** | attn reads m_{l-1}(t), m_{l-1}(t-1) (drop a_{l-1} read): -0.0004, NEW BEST |
| 52 | r52b-shift2 | r51d | 0.93657 | + attn reads a_{l-1}(t-1): no |
| 52 | r52c-shift-mlp | r51d | 0.93585 | + MLP reads a_l(t-1): -0.0006 |
| 52 | r52d-shift-gate37 | r51d | 0.93588 | + gate (lowrank 3,7): -0.0006 |
| 53 | r53a-best-mlp-shift-swap | r52a | **0.93526** | MLP reads a(t), a(t-1) instead of a(t), m_{l-1}(t): NEW BEST |
| 53 | r53b-gate37-mlp-shift-swap | r52d | 0.93539 | same on gate(3,7) variant |
| 53 | r53c-shift-docmask | r51d | 0.93669 | zero shifted read at doc starts: no |
| 53 | r53d-shift-dilated | r51d | 0.93618 | + read at t-2: -0.0003 (no) |
| 54 | r54a-gate37-k-shift-only | r54b | 0.93555 | shifted read only into keys: ~same |
| 54 | r54b-best-gate37 | r53a | 0.93568 | lowrank only layers 3,7: ~same (67.4 ms) |
| 54 | r54c-r51d-k-shift-only | r51d | 0.93612 | keys-only shift on r51d: -0.0004 |
| 54 | r54d-va-vshift | r43a | 0.94669 | previous value as value source: no |
| 55 | r55a-g37-ynorm | r54b | 0.93806 | norm(y)*gate: +0.0024 worse |
| 55 | r55b-g37-mlp-mread | r54b | 0.93528 | + MLP reads m_prev: ~best |
| 55 | r55c-kshift-attn-aread | r54a | 0.93528 | + attn reads a_prev: ~best |
| 55 | r55d-va-deep-normmix | r49b | 0.94517 | deep embed with normalized gated mixing: worse |
| 56 | r56a-kshift-x0prev | r54a | 0.93546 | + x0[t-1] in keys: no |
| 56 | r56b-g37-kshift-mread | r54a | **0.93499** | key-only shift + MLP reads m_prev: NEW BEST (marginal) |
| 56 | r56c-r51d-q-shift-only | r51d | 0.93673 | shift read only into queries: worse than keys-only (0.93612) |
| 56 | r56d-r51d-kshift-x0prev | r54c | 0.93644 | + x0[t-1] in keys: no |
| 57 | r57a-cheap-k-mshift2 | r54c | 0.93654 | keys read m(t-2): no |
| 57 | r57b-cheap-mread | r54c | 0.93605 | MLP reads a(t-1): ~same |
| 57 | r57c-k-ashift-only-cheap | r54c | 0.93585 | keys also read a(t-1): -0.0003 |
| 57 | r57d-head-mshift | r54c | 0.93713 | head reads m_L(t-1): +0.001 worse |
| 58 | r58a-best-gates-adam | r56b | **0.93404** | gate matrices labelled "other" (AdamW): -0.001 AND 3.5 ms faster (65 ms) |
| 58 | r58b-va-gates-adam | r43a | 0.94618 | same on r43a: -0.0007, 58 ms |
| 58 | r58c-cheap-kashift-gate37 | r57c | 0.93497 | (gates other) |
| 58 | r58d-va-selfv | r43a | 0.94576 | self-value skip: -0.0011 |
| 59 | r59a-lr-all-other | r58a | **0.93278** | lowrank gate in ALL layers (its matrices AdamW): -0.0013 (67.7 ms) |
| 59 | r59b-kashift-a | r58a | 0.93412 | keys read a(t-1): no |
| 59 | r59c-selfv | r58a | 0.93287 | self-value skip: -0.0012 |
| 59 | r59d-mlp-outgate | r58a | 0.93335 | MLP write *= 2sig(UVu) per channel (gate from input): -0.0007, +2.8 ms |
| 60 | r60a-vag-selfgate-lr | r58b | 0.94412 | selfgate+lowrank (AdamW): -0.0021, +5.6 ms |
| 60 | r60b-vag-mlp-outgate | r58b | 0.94454 | MLP write gate: -0.0016, +2.9 ms |
| 60 | r60c-vag-qfeat | r58b | 0.94494 | query feature gate: -0.0012, +2.3 ms |
| 60 | r60d-va-perdim-vgates | r58b | 0.94592 | per-dim value gates: -0.0003 (no) |
| 61 | r61a-lrodd-outgate-late | r59a | **0.93223** | lowrank odd layers + MLP write gates layers 4-7: NEW BEST, 67.8 ms |
| 61 | r61b-lrodd-qfeat | r59a | 0.93319 | lowrank odd + qfeat: worse |
| 61 | r61c-lr37-outgate-all | r59a | 0.93349 | lowrank 3,7 + write gates all: worse |
| 61 | r61d-vag-mlp-gate-from-attn | r58b | 0.94717 | MLP write gate from attention output: +0.001 (gate must come from own input) |
| 62 | r62a-outgate-2to7 | r61a | 0.93252 | write gates layers 2-7: ~same |
| 62 | r62b-wgate-attn-instead | r61a | 0.93330 | write gates on attention instead of MLP: worse |
| 62 | r62c-vag-attn-wgate | r58b | 0.94508 | attention write gate from input: -0.0011 |
| 62 | r62d-vag-both-wgates | r60b | 0.94405 | MLP + attention write gates: -0.0005 vs MLP only |
| 63 | r63a-attn-wgate-67 | r61a | 0.93261 | + attention write gates layers 6-7: no |
| 63 | r63b-r61a-repeat | r61a | **0.93208** | REPEAT of r61a (0.93223): confirmed, mean 0.93216 |
| 63 | r63c-vag-mlp-ingate | r58b | 0.94439 | MLP INPUT gate (u *= 2sig(UVu)): -0.0018 |
| 63 | r63d-vag-x0-chgate | r58b | 0.94479 | per-channel x0 re-injection gate: -0.0014 |
| 64 | r64a-ingate-instead | r61a | **0.93169** | MLP INPUT gates (layers 4-7) instead of write gates: NEW BEST |
| 64 | r64b-vag-ingate-x0gate | r63c | 0.94242 | + per-channel x0 gate: -0.002 more (additive), +4.6 ms |
| 64 | r64c-vag-in-out-gates | r63c | 0.94407 | MLP input + write gates: not additive |
| 64 | r64d-vag-attn-ingate | r58b | 0.94343 | ATTENTION INPUT gate: -0.0028 |
| 65 | r65a-nolr-attn-ingate-late | r64a | 0.93265 | lowrank -> attn input gates (4-7): worse |
| 65 | r65b-vag-attn-ingate-from-x0 | r64d | 0.94429 | attn input gate computed from x0: worse than from input |
| 65 | r65c-attn-ingate-instead | r64a | 0.93371 | MLP input gates -> attn input gates: worse |
| 65 | r65d-vag-attn-mlp-ingates | r64d | 0.94228 | attn + MLP input gates: -0.0012 more |
| 66 | r66a-r64a-repeat | r64a | 0.93179 | REPEAT of r64a (0.93169): robust, mean 0.93174 |
| 66 | r66b-vag-normhead | r58b | 0.95232 | LM-head rows normalized at use: +0.006 WORSE (row norms = frequency prior) |
| 66 | r66c-best-head-ingate | r64a | 0.93189 | head input gate: ~same |
| 66 | r66d-vag-value-ingate | r58b | 0.94563 | value-only input gate: -0.0006 (q/k carry the attn input-gate gain) |
| 67 | r67a-vag-qk-ingate | r58b | 0.94442 | q/k-only input gate: -0.0018 |
| 67 | r67b-vag-hidden-gate | r58b | 0.94481 | low-rank GLU on MLP hidden: -0.0014 |
| 67 | r67c-vag-prenorm-gate | r58b | 0.94602 | gate before the MLP norm: ~0 |
| 67 | r67d-vag-mlp-ingate-from-attn | r58b | 0.94721 | MLP input gate from attention output: +0.001 (cross-gating hurts) |
| 68 | r68a-best-rebalance | r64a | 0.93223 | lowrank 1,3,5 + MLP ingates 3-7: no gain |
| 68 | r68b-vag-selfgated-values | r58b | 0.94435 | self-gated values: -0.0018 |
| 68 | r68c-vag-mra-selfgated | r58b | 0.94489 | self-gated MLP read of attention: worse than plain read |
| 68 | r68d-vag-mra-plain | r58b | 0.94406 | plain MLP read of attention: -0.0021, +0.9 ms |
| 69 | r69a-best-selfv-instead | r64a | 0.93336 | self-gated values instead of MLP ingates: worse |
| 69 | r69b-vag-selfgated-q | r58b | 0.94514 | self-gated queries: -0.001 |
| 69 | r69c-vag-selfgated-k | r58b | 0.94497 | self-gated keys: -0.0012 |
| 69 | r69d-vag-qk-mlp-ingates | r67a | 0.94303 | q/k + MLP input gates: -0.0014 more |
| 70 | r70a-best-no-doctemp | r64a | 0.93319 | ablation: doc temp worth ~0.0015 |
| 70 | r70b-best-no-x0gate | r64a | 0.93305 | ablation: x0 gate worth ~0.0013 |
| 70 | r70c-best-no-kshift | r64a | 0.93306 | ablation: key-shift read worth ~0.0013 |
| 70 | r70d-vag-shared-ingate | r65d | 0.94310 | one shared attn/MLP input gate: worse than separate (+0.0008) |
| 71 | r71a-best-ingate-early | r64a | 0.93590 | MLP input gates in layers 0-3 instead of 4-7: MUCH WORSE (+0.0042) |
| 71 | r71b-best-lr-even | r64a | 0.93210 | low-rank head gate in even layers: ~same/slightly worse |
| 71 | r71c-vag-mlp-hnorm | r58b | 0.94416 | MLP hidden RMS-normalized before c_proj: -0.0020 vs r58b, ~free (58.5 ms) |
| 71 | r71d-vag-temporal-ingate | r63c | 0.94226 | MLP input gate also sees u[t-1] (extra 512->32): -0.0021 vs r63c (+1.6 ms) |
| 72 | r72a-best-hnorm | r64a | 0.93265 | + MLP hidden norm on best: +0.0010 WORSE |
| 72 | r72b-best-temporal-ingate | r64a | 0.93342 | cheap temporal ingate (shifted rank-32 feats), layers 5-7: +0.0017 worse |
| 72 | r72c-vag-temporal-hnorm | r71d | 0.94586 | temporal ingate + hnorm: +0.0036 vs r71d (hnorm conflicts with input gates) |
| 72 | r72d-vag-mlp-reads-ve | r58b | 0.94597 | MLP reads normalized VE row: ~0 |
| 73 | r73a-best-no-lowrank | r64a | 0.93254 | ablation: low-rank head gate worth ~0.0009 (saves 1.6 ms) |
| 73 | r73b-nolr-temporal-ingate | r64a | 0.93499 | lowrank -> temporal MLP ingate (r71d form): +0.0025 vs r73a. TEMPORAL GATE HURTS ON BEST |
| 73 | r73c-lr-to-selfgated-k | r64a | 0.93322 | lowrank -> self-gated keys: worse |
| 73 | r73d-best-rope-per-window | r64a | **0.93151** | RoPE base = 10000*w_h/1024 per multi-scale head: -0.0002 (noise-level), free. NEW BEST |
| 74 | r74a-doc-outtemp | r73d | 0.93134 | + f_doc in output temperature: -0.0002 |
| 74 | r74b-rope-scale-full | r73d | **0.93121** | full layers RoPE base 20000 (base ~ 10*window everywhere): -0.0003. NEW BEST |
| 74 | r74c-ve1-prevtok | r73d | 0.93453 | layer-1 VE indexed by previous token: +0.003 BAD (values need own identity) |
| 74 | r74d-doc-x0gate | r73d | 0.93141 | + f_doc in x0 gate: -0.0001 |
| 75 | r75a-rope-doc-both | r74b | 0.93116 | + doc out-temp + doc x0 gate: ~0 |
| 75 | r75b-doc-ve-gate | r74b | **0.93102** | f_doc in VE gate: -0.0002 (noise-level). NEW BEST |
| 75 | r75c-ms-layer3-rope | r74b | 0.93216 | layer 3 multi-scale (256..2048) + matched RoPE: +0.001 worse |
| 75 | r75d-rope-base-x2 | r74b | 0.93103 | RoPE bases doubled (20 x window): -0.0002 (bigger base OK) |
| 76 | r76a-docve-base-x2 | r75b | 0.93113 | doc VE gate + RoPE x2: no stacking |
| 76 | r76b-token-ingate | r75b | 0.93087 | TOKEN TABLE (8192x128) adds rank-32 features to MLP input gates 4-7: -0.00015 |
| 76 | r76c-token-headgate | r75b | **0.93074** | per-token per-(layer,head) gate bias table (8192x32): -0.0003. NEW BEST |
| 76 | r76d-all-doc-x2 | r75a | 0.93101 | all doc-position bits + RoPE x2: ~0 |
| 77 | r77a-token-controls | r76c | 0.93068 | token bias on head gates + q temps + x0 gates + out temp (73/token): ~-0.0001 |
| 77 | r77b-token-head-mlp | r76c | **0.93066** | token head-gate bias + token MLP-gate features (layers 5-7), one 128-wide table: ~-0.0001 |
| 77 | r77c-r76c-repeat | r76c | 0.93084 | REPEAT: r76c mean 0.93079 (vs r64a mean 0.93174: lineage gain ~-0.001 is real) |
| 77 | r77d-bigram-headgate | r76c | 0.93082 | + prev-token head-gate bias: ~0 |
| 78 | r78a-token-reads | r77a | 0.93141 | token biases on the 5 read strengths: +0.0007 worse |
| 78 | r78b-vag-full-docmask | r58b | 0.94557 | CONTROL: doc masking in ALL layers (known): -0.0006 |
| 78 | r78c-r76c-token-vegates | r76c | 0.93074 | token biases on VE/proj gates: ~0 |
| 78 | r78d-vag-local-docmask | r58b | 0.94428 | DOC MASKING ONLY IN LOCAL (sliding) LAYERS: -0.0019 (3x the all-layer mask) |
| 79 | r79a-lr57-local-docmask | r77b | 0.93154 | best, lowrank 5,7 + local docmask (0-2,4-6): worse |
| 79 | r79b-nolr-local-docmask | r77b | 0.93136 | best, no lowrank + local docmask: +0.0007 (lowrank worth 0.0009 -> docmask ~-0.0002) |
| 79 | r79c-vag-docmask-early | r78d | 0.94340 | DOC MASK ONLY EARLY LOCAL LAYERS 0-2: -0.0028 vs r58b (better than all local) |
| 79 | r79d-vag-docmask-late | r78d | 0.94743 | doc mask only late local layers 4-6: +0.0013 WORSE than none |
| 80 | r80a-best-docmask012-lr57 | r77b | 0.93031 | best + docmask 0-2, lowrank only 5,7: -0.0004 |
| 80 | r80b-best-docmask01-lr357 | r77b | **0.93019** | best + docmask 0-1, lowrank 3,5,7: -0.0005. NEW BEST |
| 80 | r80c-vag-docmask-0123 | r79c | 0.94441 | fast: docmask 0-3 incl. global L3: worse than 0-2 (+0.001) |
| 80 | r80d-vag-docmask-01 | r79c | 0.94453 | fast: docmask 0-1 only: worse than 0-2 (+0.0011) |
| 81 | r81a-dm012-lr357 | r80b | **0.93000** | best + docmask 0-2, lowrank 3,5,7: NEW BEST |
| 81 | r81b-dm012-lr1357-ig567 | r80b | 0.93048 | lowrank 1,3,5,7 but MLP ingates 5-7: worse (ingate L4 > lowrank L1) |
| 81 | r81c-vag-dm012-doctemp | r79c | 0.94189 | fast: docmask 0-2 + doc query temp: -0.0015 more (additive). NEW CHEAP BASE (59 ms) |
| 81 | r81d-vag-localmask-sink | r78d | 0.94410 | all-local mask with row BOS visible: ~r78d -> late layers use other docs' CONTENT, not a sink |
| 82 | r82a-dm012-shift-isolated | r81a | 0.93001 | + shifted reads zeroed at doc starts in 0-2: ~0 |
| 82 | r82b-r81a-repeat | r81a | 0.93018 | REPEAT: r81a mean 0.93009 |
| 82 | r82c-vag-dm012-late-halfheads | r81c | 0.94338 | late local layers: heads 0-1 masked: +0.0015 worse (all late heads need other docs) |
| 82 | r82d-vag-lexical-match | r81c | 0.94152 | q,k += a_h/b_h * norm(x0 head slice): -0.0004, but +5.7 ms |
| 83 | r83a-vag-dm-ingate | r81c | 0.94006 | fast + MLP ingates 4-7: -0.0018. NEW CHEAP BASE (60 ms) |
| 83 | r83b-best-lex-7 | r81a | 0.93021 | lexical match in layer 7 only: ~0 |
| 83 | r83c-vag-late-full | r81c | 0.94177 | late layers full context (no window): ~0 (and faster on fast base) |
| 83 | r83d-vag-doc-jump-rope | r81c | 0.94237 | RoPE position += T per document: +0.0005 worse, +5 ms |
| 84 | r84a-best-free-tweaks | r81a | 0.93009 | + RoPE x2 + doc out temp: 0 |
| 84 | r84b-noshift012-lr1357 | r81a | **0.92944** | no key-shift read in isolated layers 0-2 + lowrank back in layer 1: -0.0007. NEW BEST |
| 84 | r84c-vag-late-prevdoc | r83a | 0.94009 | late local layers see only current + previous doc: = r83a (the NEIGHBOUR doc is what they use) |
| 84 | r84d-vag-early-nope | r83a | 0.97885 | NoPE in isolated early (windowed) layers: CATASTROPHIC +0.039 |
| 85 | r85a-noshift-all012-ig3 | r84b | 0.93102 | + no MLP a[t-1] read in 0-2 + MLP ingate L3: worse |
| 85 | r85b-r81a-noshift012 | r81a | 0.93005 | r81a - key shift in 0-2: ~same (neutral) |
| 85 | r85c-noshift-all012 | r84b | 0.93015 | r84b - MLP a[t-1] read in 0-2: +0.0007 vs r84b (or r84b lucky) |
| 85 | r85d-vag-early-docfull | r83a | 0.94029 | early layers whole doc, no window: ~same -> ISOLATION matters, not locality |
| 86 | r86a-r84b-repeat | r84b | 0.92960 | REPEAT: r84b mean 0.92952 (real gain vs r81a mean 0.93009) |
| 86 | r86b-r84b-rope2 | r84b | **0.92899** | + RoPE bases x2 (20 x window): -0.0005. NEW BEST |
| 86 | r86c-r84b-doc-out-x0 | r84b | 0.92930 | + doc pos in out temp and x0 gate: -0.0002 |
| 86 | r86d-vag-prevdoc-summary | r83a | 0.94007 | late layers read previous doc's last state: 0 |
| 87 | r87a-rope2-doc-out | r86b | **0.92888** | + doc pos in output temperature: -0.0001. NEW BEST |
| 87 | r87b-rope4 | r86b | 0.92905 | RoPE bases x4: saturates |
| 87 | r87c-vag-ingate-all | r83a | 0.94019 | MLP ingates all layers: = late-only |
| 87 | r87d-vag-ingate-early | r83a | 0.94132 | MLP ingates early only: -0.0006 vs none, worse than late |
| 88 | r88a-dm-spare-long-head | r87a | 0.92996 | early mask spares the 1024 head: +0.001 WORSE (all early heads want isolation) |
| 88 | r88b-token-qtemp | r87a | 0.92911 | token q-temp biases instead of token MLP-gate feats: ~same |
| 88 | r88c-vag-doc-parity | r83a | 0.93994 | late layers: q,k += parity(doc) * learned dirs: -0.0001 (+6 ms) |
| 88 | r88d-vag-docstart-prior | r83a | 0.94014 | vocab bias * (1 - f_doc): 0 |
| 89 | r89a-r87a-repeat | r87a | **0.92878** | REPEAT: r87a mean 0.92883 |
| 89 | r89b-early-rope-half | r87a | 0.92914 | RoPE base 10 x w in isolated layers: slightly worse |
| 89 | r89c-vag-dm0124 | r83a | 0.93953 | fast: mask 0-2 AND 4: -0.0005 vs 0-2 (boundary is not exactly at layer 3) |
| 89 | r89d-doc-headgate | r87a | 0.92886 | + f_doc term in head gates: ~0 |
| 90 | r90a-r86b-dm0124 | r86b | 0.92901 | + mask layer 4: 0 |
| 90 | r90b-vag-dm01245 | r89c | 0.94058 | fast: mask 0-2,4,5: worse |
| 90 | r90c-vag-dm012456 | r89c | 0.94068 | fast: mask all local: worse than 0-2 (r89c's L4 gain probably noise) |
| 90 | r90d-dm0124-lr357 | r87a | 0.93008 | mask L4 but drop lowrank L1: +0.0012 (lowrank L1 is valuable) |
| 91 | r91a-bd-lr13567 | r87a | 0.92985 | block-diag lowrank form + lowrank in L6: +0.001 worse (?) |
| 91 | r91b-vag-doc-title-x0 | r83a | 0.93999 | late layers re-inject doc's first token: 0 |
| 91 | r91c-vag-headgate | r83a | 0.93837 | cheap base + head self-gate (scalar + bd low-rank) all layers: -0.0017 |
| 91 | r91d-vag-yhead-norm | r83a | 0.93883 | PER-HEAD ATTENTION OUTPUT NORMALIZED (x learned per-head scale): -0.0012 |
| 92 | r92a-best-ynorm | r87a | 0.93315 | y-norm WITHOUT per-head scale on best: +0.0043 BAD |
| 92 | r92b-vag-headgate-ynorm | r91c | 0.93737 | head gate + y-norm(scale): -0.001 more (stacks) |
| 92 | r92c-vag-ynorm-noscale | r91d | 0.94113 | y-norm without scale: +0.0023 vs with scale -> PER-HEAD SCALE ESSENTIAL |
| 92 | r92d-vag-ynorm-late | r91d | 0.93970 | y-norm late layers only: -0.0004 (a third of all-layer gain) |
| 93 | r93a-best-yn-early | r87a | 0.93027 | best + y-norm(scale) layers 0-3 (bd form): +0.0014 worse |
| 93 | r93b-best-yn-late | r87a | 0.93049 | ... layers 4-7: +0.0017 worse |
| 93 | r93c-best-yn-even | r87a | 0.93109 | ... even layers: +0.0023 worse -> y-norm does NOT transfer to the best |
| 93 | r93d-vag-write-norm | r91d | 0.94352 | norm(c_proj y) * s_l: +0.005 BAD |
| 94 | r94a-best-blockdiag | r87a | **0.92868** | r87a with the low-rank head gate as block-diagonal matmuls (same function, 67.9 ms): NEW BEST |
| 94 | r94b-bd-doc-x0 | r87a | 0.92903 | bd + doc term in x0 gate: ~0 |
| 94 | r94c-vag-ynorm-learned-power | r91c | 0.93782 | y / rms^a_h learned: -0.0006 (less than full norm + scale) |
| 94 | r94d-vag-ynorm-early | r91d | 0.93831 | y-norm(scale) early layers 0-3 only: -0.0018 vs r83a (better than all layers) |
| 95 | r95a-r94a-repeat | r94a | 0.92905 | REPEAT: r94a mean 0.92887 (= r87a family) |
| 95 | r95b-bd-lr0137 | r94a | 0.93015 | + lowrank L0: +0.0015 worse |
| 95 | r95c-vag-hg-ynorm-early | r92b | 0.93728 | head gate + y-norm 0-3: = all-layer y-norm (r92b) |
| 95 | r95d-vag-ynorm-012 | r94d | 0.93989 | y-norm only in isolated layers 0-2: ~0 -> the 0-3 gain comes from GLOBAL LAYER 3 |
| 96 | r96a-best-yn-3 | r94a | 0.92952 | best + y-norm layer 3: +0.0007 |
| 96 | r96b-best-yn-37 | r94a | 0.92921 | best + y-norm layers 3,7: +0.0003 |
| 96 | r96c-vag-yn-3 | r94d | 0.93924 | fast: y-norm layer 3 only: -0.0008 |
| 96 | r96d-vag-yn-37 | r94d | 0.93936 | fast: y-norm layers 3,7: -0.0007 |
| 97 | r97a-abl-no-docmask | r94a | 0.93027 | ABLATION: early doc masking worth 0.0014 on the final model |
| 97 | r97b-abl-no-token-headgate | r94a | 0.92916 | ABLATION: token head-gate table worth 0.0003 |
| 97 | r97c-abl-rope-10000 | r94a | 0.92967 | ABLATION: RoPE base per window worth 0.0008 |
| 97 | r97d-vag-early-no-bos | r83a | 0.94084 | isolated layers without their doc's BOS key: +0.0008 (BOS is their sink) |
| 98 | r98a-vag-early-rowbos | r83a | 0.93995 | early mask + row BOS visible: ~0 |
| 98 | r98b-best-early-rowbos | r94a | 0.92887 | same on best: 0 |
| 98 | r98c-best-qkgate-7 | r94a | 0.92976 | + q/k input gate L7: +0.0009 worse |
| 98 | r98d-best-qkgate-57 | r94a | 0.93010 | + q/k input gate L5,L7: +0.0012 worse |
| 99 | r99a-vag-global-prevdoc | r84c | 0.94033 | fast: global layers cur+prev doc only: ~same (+0.0003) |
| 99 | r99b-vag-all-prevdoc | r99a | 0.94031 | fast: all later layers cur+prev doc only: ~same -> docs beyond the neighbour barely matter |
| 99 | r99c-r94a-repeat2 | r94a | 0.92882 | REPEAT 2: final model mean 0.92885 (3 runs: 0.92868, 0.92905, 0.92882) |
| 99 | r99d-vag-early-parity-soft | r88c | 0.94260 | learned parity bias instead of hard early mask: +0.0025 vs hard mask (cannot learn isolation) |
| 100 | r100a-vag-rope-window | r92b | 0.93622 | TRANSFER: RoPE base ~ window on cheap base: -0.0012 |
| 100 | r100b-vag-token-headgate | r92b | 0.93641 | TRANSFER: token head-gate table: -0.0010 |
| 100 | r100c-vag-abl-docmask | r92b | 0.93997 | TRANSFER: no early doc mask: +0.0026 |
| 100 | r100d-vag-multiscale | r92b | 0.94068 | multi-scale windows on cheap base: +0.0033 WORSE |
| 101 | r101a-vag-rope-token | r100b | 0.93567 | RoPE ~ window + token head gates stack on cheap base: -0.0017 vs r92b |
| 101 | r101b-vag-doc-outtemp | r101a | 0.93545 | + doc-position output temperature: -0.0002 |
| 101 | r101c-vag-kshift | r101a | 0.93640 | + key reads u[t-1] (non-isolated layers): +0.0007 worse (without the MLP-output chain) |
| 101 | r101d-abl-token-mlp | r94a | 0.92907 | ABLATION: token MLP-gate features worth ~0.0002 |

## Learned
- R1: all three stable ideas gave small gains (0.001-0.0024). Self-gating of attention output is best.

- R2: bounded per-token query temperature works (exp version diverged). Multi-scale per-head windows are a
  strong, cheap win (also faster). ve-keys redundant once heads self-gate.

- R3: multiscale windows and query temperature overlap (both control locality/sharpness). Key-side
  temperature hurts. Self-gating only works where the gated quantity is bounded (attention output).
  Principle that worked: a gainless norm removes a magnitude channel; restoring a bounded per-token scale
  at the query helps.

- R4: two more "bounded per-token scalar gate from unit-RMS input" ideas work on baseline (out-temp, x0gate).

- R5: NOISE between identical runs ~0.0003 -> treat |diff| < 0.0005 as noise. Gains on top of the best
  model are getting small (<0.001). Out-temp does not stack. MLP-scale gates hurt (ReLU^2 is 2-homogeneous).

- R6: SOURCE-side per-token scales (key salience r03c, value strength r06a) HURT; query/output-side help.
  Induction shortcut saturates at 1 prev token.

- R7: nothing stacks on r05b. flex_attention score_mod capturing a learnable tensor CRASHES the 8-GPU speed
  benchmark (no error shown) -> avoid score_mod with grads. Never rescale the carried residual per token.

- R8: richer self-gate (per-dimension, from the head's own output) is the most productive direction.
  Forcing competition between heads is very bad. Full 512x512 gate exceeds attention param cap (2.1M).
  Einsum-based gates cost ~4 ms; q-aware gate (extra einsum) exceeded step cap.

- R9: per-head structure of the gate matters (cross-head worse). Gate rank saturated at 32. MLP benefits
  from an undiluted x0 read. Lit check: no prior gate computed from the attention output itself (Qiu 2025
  gates from input x).

- R10: "undiluted read" of the block's own attention output by the MLP is a big base win. Per-dim query
  gate helps a bit. Time budget is the binding constraint for the best model (r10b 68.6 ms).
- Practical: Bash foreground timeout max 600 s — use run_in_background for check waits too.

- R11: mra stacks on the gated model (-0.0022). Related lit: Attention Residuals (Kimi 2026, depth-wise
  softmax over all previous layers), Multi-Gate Residuals (2026). Ours: one scalar, MLP reads its block's
  fresh attention output re-normalized.

- R13: the "undiluted read" chain keeps giving: attn<-m_{l-1}, MLP<-a_l, MLP<-m_{l-1}. Timing (noisy):
  prev-token keys ~1.4 ms, x0 gate ~0.2 ms; matmul vs einsum for the gate: no difference.

- R14: per-token scalar multipliers on MLP/residual writes are consistently BAD (r03d, r07a, r14d); only the
  attention output (per head, pre-projection) tolerates/benefits from gating. Low-rank per-channel gate on
  MLP output cost ~5 ms (71.7) — elementwise on [B,T,512] in fp32 is expensive.

- R20: doc-position temperature substitutes for the content q-temperature (no stacking). Plateau ~0.9444 for
  the r14c family, which sits at the time cap.

- R25: plateau ~0.9444 from two lineages (r20b qtemp family; r25a cheap doc-temp family).

## Best so far
r28a-best-outtemp 0.94335 (r26a + content output temperature)
r26a-vegate-full 0.94391 (r25a + VE gate from full input)
r25a-cheap-de-lowrank-half 0.94441 (doctemp + scalar&half-lowrank gate + chain + ms + x0gate + deep-embed)
r20b-best-doctemp 0.94439 (= r14c + doc term in qtemp; tie with r14c)
r14c-chain3-attn-reads-attn 0.94455 (4 undiluted reads, no prev-token keys)
r13c-mlp-reads-prevmlp 0.94522 (chain of 3 undiluted reads)
r12b-best-attn-reads-mlp 0.94688 (r11a + attention reads previous MLP output)
r11a-ms-mra 0.94787 (r09a + MLP reads attention)
r10b-qfeat-gate 0.94946 (r09a + per-dim query gate)
r09a-gate-multiscale 0.95006 = baseline + attn scalar+lowrank selfgate + q temp + prev-token keys + x0 gate
  + multiscale S-layer head windows (66 ms)

## Plan
R16: ALL failed step cap (rank-16 gate + prevtok 69.8, +qfeat 71.2, bilinear read 69.1, split MLP 71.9).
TIMING NOISE IS LARGE (+-1.5 ms): r14c measured 68.3 but r14c minus qtemp 69.2, minus scalar gate 69.9,
per-channel 72.7. The r14c family's true time is ~69 -> additions on it mostly fail. Gate as matmuls (bf16)
gives no speedup. Low-rank gate costs ~4.5 ms (r05b 63.7 -> r08b 68.2). Step is matmul-bound (~48 ms);
lm_head is 14% of matmul FLOPs.
- TIME LESSON: per-channel 512-vector params multiplying [B,T,512] activations are EXPENSIVE (~1+ ms each,
  gradient reduction): per-channel chain weights 72.7/79.8 ms. Prefer scalar weights; per-channel prevtok
  lambda probably costs ~1.4 ms. gather on [B,T,512] also expensive (doc-bag 69.8).
- SPEED QUIRK: adding the scalar head self-gate makes models FASTER (r20a 66.0 -> r21b 62.5; base 66 ->
  r01b 61). Probably memory layout of the attention output/grad. Without it the chain measured 70.7.
- R32: torch.cumsum-based features failed the COMPILED causality check (numerical leak 0.09) in r28a; use a
  strictly lower-triangular matmul for prefix sums (exactly causal). Novelty adds nothing on r28a.
  Moving out-temp before the head (linear) did not measurably save time. No soft-cap saves ~0.7 ms.
- R35: VALUE NORMALIZATION stacks strongly on the chain model (-0.0023): values as unit directions; output
  magnitude = agreement. (Source-side magnitude *adding* hurt; removing it helps.)
- R36: normalizing the VE separately (so the gate sets the exact mix) is a further big gain.
- R39: the gated normalized value path is a HUGE general gain (-0.0102 on r01b). Baseline VE tables
  (Adam, lr 1e-2) probably grow and dominate values; normalizing each value source and mixing with gates fixes it.
  Many earlier conclusions may have been confounded by this. r39d is a cheap new base (63 ms).
- x0 as value source in non-VE layers measured 81 ms (pathological), dropped.
- R43: MECHANISM: VE tables are sparse (row updated only when its token appears) and start at ~1/sqrt(512) of
  the projected values' scale -> RMS-normalizing VE at use (one line) gives -0.011 on the baseline.
  Principle: sparse-update embedding tables should be normalized where they are used (wte already is).
- Hypothesis for the VE problem: AdamW weight decay (0.1) hits every VE row every step but gradients reach a
  row only when its token occurs -> rare-token rows shrink; plus tiny init. Normalization at use fixes both.
- R50: r50c (value path + deep + doc/out temps + ms + x0gate + full chain, NO head self-gate) = best level at
  63 ms -> new cheap base with ~5 ms headroom.
- R58: GATES SHOULD BE "other" (AdamW), as CLAUDE.md lists: Muon on many tiny gate matrices is slow AND worse.
- R60 attempts on r59a all failed the step cap (69-71 ms): r59a has no headroom. The self-value skip costs ~3 ms
  (unexpectedly). Moved gate re-measurements to the fast r58b base (value path + AdamW gates, 58 ms).
- R63-64: INPUT-SIDE PER-CHANNEL GATES (feature selection from the sublayer's own normalized input, low-rank,
  AdamW) are a new productive family: attention input -0.0028, MLP input -0.0018, x0 channel gate -0.0014.
- Timing survey on r64a (67.8 ms): -deep embed 67.7, -lowrank(odd) 66.2, -MLP input gates 65.6, -out temp 68.7
  -> gates cost ~0.4 ms/layer; deep embed ~free.
- R67: SELF-gating (gate computed from the tensor it gates) works; CROSS-gating (from another signal) hurts.
- R71: two new wins on fast base: MLP HIDDEN NORMALIZATION (norm(relu^2) before c_proj) and TEMPORAL input gate.
  Input gates belong in LATE layers (early is +0.004). (keys reading norm(x)[t-1] failed cap 69.3.)
- R72: fast-base wins did NOT transfer: hnorm hurts on gated models; cheap temporal gate worse. Multi-scale
  windows in layer 3 (timing probe) saves no time (68.1).
- R73: the best already has local context (key shift, a[t-1] reads), so temporal gates are redundant/harmful there.
- R74: RoPE base proportional to receptive field: two small consistent steps (0.93174 -> 0.93151 -> 0.93121).
  Learnable per-head RoPE strength (cos + c_h) costs ~2 ms (grad reduction over [B,T,H,D]) -> fails cap.
- R75: all within noise. Layer 3 needs full-context heads.
- R76: token-identity tables steering gates (cheap: lookups, embedding cap has 1.05M spare) give small gains.
- R77: token tables saturate quickly (~0.0001 more). Doc-mask score_mod costs ~1.3 ms on the best (fails cap).
- R78: LOCAL-ONLY DOCUMENT MASKING: local layers want the current document, global layers benefit from other
  documents (all-layer masking gives only a third of the gain). score_mod costs ~1.3 ms; per-batch create_block_mask
  FAILS the precision check (int8 inside).
- R79: DEPTH SPLIT: early layers want document isolation, late layers want cross-document context.
- R80: mask = the LOCAL layers of the FIRST HALF (0-2); the global layer 3 should see other documents.
Budget exhausted after R101 (400 experiments). Final best: r94a-best-blockdiag 0.92868 (mean of 3 runs 0.92885).
R100: transfer tests on cheap base r92b: RoPE ~ window; token head-gate table; no docmask; multi-scale windows.
R99: fast: global layers cur+prev doc; all later layers cur+prev doc; soft parity isolation early (no hard mask); r94a repeat 2.
  (lowrank rank 16 in all layers: 69.5 fail)
R98: early mask + row BOS visible (fast, best); best + q/k input gate in L7 / L5,L7. (Sink-only trick via flex return_lse
  FAILS torch.compile: no backward through lse.)
R97: ablations of the final model (no docmask / no token head-gate table / RoPE 10000); fast: isolated layers without BOS key.
R96: y-norm only in global layer 3 / layers 3+7, on best and cheap base.
R95: r94a repeat; bd + lowrank L0; fast: head gate + early y-norm; y-norm 0-2 only.
R94: bd form alone on best (equivalence check); bd + doc x0 gate; fast: learned partial y-norm power; y-norm early only.
R93: best + y-norm(scale) in layers 0-3 / 4-7 / even (all-layer 69.2 fail); fast: write normalization (norm(c_proj y) * s_l).
R92: best + y-norm (bd gate form, no scale: 68.1; with scale 69.2); fast: head gate + y-norm; y-norm no scale; y-norm late only.
R91: BLOCK-DIAGONAL low-rank gate (F.linear with torch.block_diag of the per-head factors; same function) saves ~0.5 ms
  (r87a 68.4 -> 67.9) -> lowrank 1,3,5,6,7 fits (68.6); all layers 69.8, +L2 69.2, +L4+L6 69.4 fail.
  Also failed: lowrank all + no key shift 70.1.
  R91 runs: bd lowrank 1,3,5,6,7; fast: doc first-token re-injection late; fast + head gates; per-head output norm.
R90: r86b + mask L4 (r87a + L4 = 68.9 fail); r87a + mask L4 w/ lowrank 3,5,7; fast: mask 0-2,4,5 / all local.
R89: r87a repeat; finer RoPE (base 10 x w) in isolated layers 0-2; fast: mask 0,1,2,4; doc term in head gates.
R87: r86b + doc out temp (with x0 doc: 69.2 fail); RoPE x4; fast: MLP ingates all layers / early only.
R86: r84b repeat; r84b+RoPE x2; r84b+doc out/x0; fast: late layers read previous doc's last state.
Time trades failing: lowrank all layers w/o token MLP feats 70.0; lowrank +L6 69.4.
R85: r84b - MLP a[t-1] read in 0-2 (+ ingate L3); ablation r81a - keyshift 0-2; fast: early layers whole doc (no window).
R84: best + RoPE x2 + doc out temp; no key shift in 0-2 + lowrank 1; fast(r83a): late layers only current+prev doc; NoPE in isolated early layers.
Lit check (2026-10): no prior found for depth-split document masking (early isolated, late cross-doc).
R83: fast+ingates (base); best+lexical L7; fast: late layers full context; doc-jump RoPE (pos += T per doc).
TIME NOTE: SDPA full attention is faster than flex sliding on the fast base (late-full 58.3 vs 59.4), but on the best
model removing multi-scale windows in 4-6 is slower (69.2). Lexical in layers 3+7 69.6; late 2x windows 69.6.
R82 plan: r81a + shift reads isolated at doc starts in 0-2; r81a repeat; fast: late layers half heads masked; lexical-match channel (64 ms).
R81 plan: best docmask 0-2 + lowrank 3,5,7; + lowrank 1,3,5,7 w/ ingates 5-7; fast: dm012 + doc temp; all-local mask w/ BOS sink visible.
R80 plan: best + docmask 0-2 (lowrank 5,7) / docmask 0-1 (lowrank 3,5,7); fast: docmask 0-3 (incl. global L3); docmask 0-1.
R79 plan: best-lowrank+local docmask (lowrank kept in 5,7 / removed); fast base: docmask early local layers only vs late only.
R78 plan: token read strengths (r77a); token VE gates (r76c); doc masking local layers only vs all layers (fast base).
R77 plan: token bias on all controls (head gates, q temps, x0 gates, out temp); head+MLP token gates (one 128 table); r76c repeat; head-gate bias from prev token too.
R76 plan: r75b+x2; token-conditioned MLP gate (E[token] 32-d into the gate, fills embedding cap); token per-head gate bias; all doc + x2.
R75 plan (on r74b): doc out-temp + doc x0 gate; doc-pos in VE gate; layer 3 multi-scale + matched RoPE; RoPE bases x2 (control).
R74 plan (on r73d): doc-pos in out temp; base 20000 for full layers; layer-1 VE by previous token; doc-pos in x0 gate.
R73 plan was: r64a-no-lowrank (ablation/base), nolr+temporal ingate (r71d form), lowrank->self-gated keys, RoPE base per head window.
R72 plan was: r64a+hnorm, r64a+temporal gate (5-7, cheap: U z_t + U' z_{t-1}), r71d+hnorm, MLP reads VE (explore).
R71 plan was: MLP ingates in layers 0-3; low-rank gate in even layers; keys read norm(x)[t-1]; temporal input gate (r63c).
- R52-53: TIME-SHIFTED READS (previous position's normalized sublayer outputs) help: a cheap local
  convolution over the previous sublayer's conclusions. Reads at t-2, doc-boundary masking: no. Tip: when an explore is too slow on baseline, build it on r01b (scalar
self-gate parent, ~61 ms) and compare to r01b (0.95653).
Known-tricks list (modded-nanogpt README): U-net skips, smear, sparse attn gate, key offset, paired heads,
backout, bigram hash emb, MUDD, partitioned hyperconnections, asym logit rescale, XSA, multi-token. Avoid
re-implementing these as-is.
