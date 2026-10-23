# Notes

## FINAL STATE (after round 101)
Best: r97b-r90b-s4 val_bpb 0.759795 (config = experiments/r90b-nonest-mu09-attn095r/train.py with SEED=4).
r90b config seed spread (21 seeds): 0.75980-0.76048, mean ~0.7601. Baseline 0.95895 (-0.199 bpb).

Final recipe (vs baseline), roughly in order of impact:
1. Scale under the 60-min wall limit (tokens fixed at 2B): 16 layers x width 1792 (14 heads x 128), MLP hidden 12800
   (~7x, ReLU^2). MLP width beat depth per unit time once compute-bound. Device batch 16 (2 micro-steps).
2. Hashed n-gram tables with a sign trick (zero init, AdamW at emb LR x7, no WD): input bigram (262139 rows) added to
   the token embedding; output-side bigram, trigram, 4-gram tables (65521 rows) added before the final norm.
3. Attention: QK-norm scale 0.12, half-truncated RoPE, window pattern SL (S = 256 sliding window), learnable XSA +
   head-wise sigmoid output gate (fused autograd), canon (causal depthwise conv k=2, bf16) before attn and MLP,
   value embeddings on alternate layers.
4. Residual routing: x0 lambdas + resid lambdas, U-net skips with dynamic gates, backout (x - lambda*x_{2L/3}),
   last 5 attention layers read mix(x_backout, x) with learnable per-layer mix (init 0) [PR #241 idea].
5. Optimizer: Muon (Polar Express NS, RMS factor 0.17, NO Nesterov, momentum warmup 0.8->0.9 over 300 steps in 5
   levels, attention matrices momentum x0.95/0.92), AdamW bf16-moment (beta1 0.9; embeddings beta1 0.85), LR 1.2e-2,
   head LR x0.4, WD 0.1 (0 for embeddings), warmup 2% then linear decay to 0.
6. Systems: bf16 residual stream, fused ReLU2Proj (recompute), bf16 Muon momentum + single bf16 all_reduce of updates,
   grad stash in bf16 between micro-steps + single flat bf16 grad all_reduce, empty_cache at step 0, no per-step host
   syncs. ~3330s train / ~3450s total (effective limit ~3575s; slow nodes add up to ~100s).

Biggest single wins (approx.): model scale (-0.11 vs baseline), MLP width regime (-0.01), n-gram tables + sign (-0.006),
XSA (-0.003), Muon speedup -> bigger model, gated attn/skips (-0.001 each), backin trick (-0.0015), momentum
schedule (warmup + per-type + no Nesterov, -0.002 total).
Dead ends: FP8, DDP, per-block compile, MoE (memory/quality), MTP, NorMuon, doc masking, factorized tables,
WSD/LR floor/decay shapes, cautious WD, GQA, value residual, extra embedding tables, removing VE/canon/resid lambdas.

## Timing / memory (8xH100)
- d8w512 (baseline): 250s, 15.5GB @ batch32. d12w768: 619s, 32GB @32. d16w1024: 1279s (0.335 s/step), 30GB @ batch16.
- Max safe training ~3400s => ~0.89 s/step.

## Results
| name | val_bpb | note |
|---|---|---|
| baseline | 0.95895 | |
| r01a-d12w768 | 0.88698 | scale |
| r01b-d16w1024 | 0.84474 | scale (best r1) |
| r01c-grouplr | 0.94869 | emb LR x5, head LR x0.4 on baseline: -0.010 |
| r01d-fullattn-veall | 0.95723 | full attn + VE everywhere: ~nothing (-0.002) |

| r02a-d16w1024-grouplr | 0.84128 | grouplr helps at scale (-0.0035); batch32 barely faster (1247s, 56GB) |
| r02b-d20w1280-grouplr | 0.81851 | 2020s train (0.53 s/step), batch16 |
| r02c-d16w1024-grouplr-lr7e3 | 0.84038 | LR 7e-3 slightly better than 1e-2 at d16 |
| r02d-attngate | 0.94888 | head-wise attn gate on r01c: nothing |

| r03a-d24w1408-lr7e3 | 0.80296 | 2826s train, 2951s total (0.74 s/step), 62GB @ batch16. ~20% headroom left |
| r03b-d20w1280-lr5e3 | 0.81467 | LR 5e-3 > 1e-2 at d20 (-0.004) |
| r03c-nesterov | 0.94802 | small: -0.0007 vs r01c |
| r03d-unet-skips | 0.94713 | small: -0.0016 vs r01c |

| r04a-d24w1408-lr5e3 | 0.80223 | LR 5e-3 vs 7e-3 at d24: -0.0007 |
| r04b-d24w1408-lr5e3-unet-nest | 0.80191 | BEST so far. unet+nesterov: -0.0003 at scale (noise?) |
| r04c-fp8-d16 | 0.84425 | FP8 linears: only 2% faster (1231 vs 1254s), +0.004 worse. DEAD END |
| r04d-d20w1280-lr3e3 | 0.81724 | LR 3e-3 worse than 5e-3 at d20 -> optimum ~5e-3 |

| r05a-d16w1792-unet-nest | 0.80037 | BEST. wider/shallower beats d24w1408 and is a bit faster (2790s, 54.5GB) |
| r05b-r04b-rerun-profile | 0.80190 | rerun = 0.801906 -> same-seed runs deterministic to 1e-5 |
| r05c-normuon | 0.95045 | NorMuon worse (+0.003 vs r03d). dead |
| r05d-smear | 0.94558 | smear: -0.0015 vs r03d |

Profile (r05b, d24w1408, 745 ms/step): matmuls ~320ms, Muon.step ~148ms of which list all_gather (NCCL LL
broadcast) ~99ms (!), grad all_reduce ~23ms, flex attn ~70ms, rest elementwise/norm/CE.
=> r06 fast Muon: flat async all_gather_into_tensor per chunk of 8 params, waits at end.

| r06a-d16w1792-smear-fastmuon | 0.79974 | smear -0.0006 at scale. fast Muon gave NO speedup (731 vs 734 ms/step) |
| r06b-d18w1792-smear-fastmuon | 0.79592 | BEST. 3189s train / 3299s total (near 3600 limit!), 57.6GB |
| r06c-d12w2048-smear-fastmuon | 0.80139 | too wide/shallow: worse than d16w1792 though faster (691ms) |
| r06d-attnscale | 0.94420 | attn scale 0.12: -0.0014 vs r05d |

Size ceiling: d18w1792 is about the max at current throughput (0.836 s/step incl. everything).

| r07a-d18w1792-attnscale | 0.79402 | BEST. attn scale 0.12 -0.0019 at scale. 3220s train |
| r07b-d18w1792-attnscale-lr4e3 | 0.79477 | LR 4e-3 slightly worse than 5e-3 |
| r07c-wd02 | 0.94546 | WD 0.2 worse (+0.0013 vs r06d) |
| r07d-canon | 0.93754 | Canon convs (k=4, before attn & MLP): -0.0067 vs r06d!! but 324s vs 263s |

r07a phase timing (synced, ms/step): forward 210, backward 428, grad allreduce 34, AdamW 2, Muon 195 (!!).
Muon NS FLOPs are only ~4 TF/rank/step (~7ms) => Muon overhead is something else; r08b times it in detail.

| r08a-d16w1792-canon | 0.79463 | canon at d16 (attnscale): ~-0.003 vs est. d16+attnscale; time +9.5% (3050s) |
| r08b-d18w1792-wd005-muontime | 0.79615 | WD 0.05 worse than 0.1 (+0.002). WD 0.1 stays |
| r08c-canon-conv1d | 0.93748 | same quality, much slower (466s). keep shifted-add canon |
| r08d-softcap30 | 0.93984 | softcap 30 worse (+0.0023). keep 15 |

r08b Muon breakdown (ms/step): NS 7168x1792 7.4, NS 1792x1792 55 (dynamic-shape recompile suspected),
gather (hidden between chunks) ~140 => all_gather ~20GB/s while grad all_reduce ~100GB/s.

| r09a-d16w1792-canon-muonfix | 0.79473 | Muon FIX WORKS: Muon 160 -> 19 ms/step; 2546s train (vs 3050), 74GB @ b16 |
| r09b-d18w1792-canon-muonfix | crash | OOM at device batch 16 (canon adds activations) -> use batch 8 |
| r09c-canon-attn-only | 0.94020 | only -0.004 (vs -0.0067 both), 309s: keep both canons |
| r09d-partial-rope | 0.93717 | -0.0004 vs r07d: neutral/slight |

Muon fix = muon_update @torch.compile(dynamic=False) + dynamo cache_size_limit 64 + each rank writes its
round-robin share of updates into a zeroed flat bf16 buffer, one all_reduce, all ranks apply all updates.
Overhead (total - train) ~115-130s; keep train <= ~3350s.

| r10a-d18w1792-canon-b8 | 0.79092 | 3016s train/3139 total; batch 8 slower than 16 (fwd 239, bwd 495, allreduce 48) |
| r10b-d20w1792-canon-b8 | 0.78787 | BEST. 3281 train / 3418 total (TOO close to 3600!) 52GB |
| r10c-d22w1792-muonfix | 0.78863 | no canon, b16: 3135 train / 3272 total, 71.5GB. depth ~= canon per unit time |
| r10d-polar-express | 0.93692 | -0.00025 vs r09d: tiny |

| r11a-d16-canon-ddp | 0.79469 | DDP+compile SLOWER (2705s vs 2546; fwd/bwd slower from graph splits). dead |
| r11b-d16-canon-bf16 | 0.79477 | bf16 canon: same loss, 2495s (-2%), 63GB (-11GB) -> keep |
| r11c-d16-canon-emb10 | 0.79494 | emb LR x10: no gain. keep x5 |
| r11d-mtp-aux | 0.94676 | MTP aux loss (t+2, w 0.3): +0.01 WORSE. dead |

Helpers: /tmp/patch_pe.py FILE (adds Polar Express + partial RoPE), /tmp/docmask.py FILE.

| r12a-d20-canonbf16-b8-pe | 0.78648 | BEST. 3213 train / 3347 total |
| r12b-d18-canonbf16-b16-pe | 0.78959 | 2801 / 2926 total |
| r12c-d16-canonbf16-win256 | 0.79463 | S-window 256 (vs 1024): -0.00014 AND -4% time (2400s). keep |
| r12d-docmask | 0.93582 | doc masking -0.0011 but +12% time at small scale (eager create_block_mask + flex for full layers) |

| r13a-d20-canon-b8-pe-win256 | 0.78705 | win256 at d20: +0.0006 (noise?) but 3087 train / 3220 total (-4%), 45GB @ b8 |
| r13b-d20-canon-b16-membudget | 0.78686 | activation_memory_budget 0.8: much slower (3430 train/3570 total!!). dead |
| r13c-d16-win256-lr6e3 | 0.79377 | LR 6e-3 > 5e-3 by 0.0009 at d16 |
| r13d-qgain | 0.93659 | per-head q gain: -0.0003 (tiny) |

| r14a-d20-win256-lr6e3 | 0.78627 | LR 6e-3 at d20: -0.0008 vs r13a |
| r14b-d21-win256-lr6e3 | 0.78373 | BEST. d21: -0.0025 vs d20 (big; odd depth / unet middle?). 3301 train / 3442 total |
| r14c-d16-win256-lr8e3 | 0.79318 | LR 8e-3 > 6e-3 (0.79377) > 5e-3 (0.79463) at d16 |
| r14d-d16-allshort-lr6e3 | 0.79513 | all-short windows worse (+0.0014), barely faster. dead |

| r15a-d21-lr8e3 | 0.78259 | BEST. LR 8e-3 at d21: -0.0011. b8: 3255 train / 3393 total, 48GB |
| r15b-d21-b16-fusedmlp-lr8e3 | 0.78273 | b16 + fused ReLU2Proj: same loss, 3162 train / 3304 total (-90s), 73GB |
| r15c-d16-lr1e2 | 0.79290 | LR 1e-2 > 8e-3 slightly (-0.0003) at d16 |
| r15d-d19-mlp3x-lr8e3 | 0.79213 | MLP3x d19: -0.001 vs d16 but +7% time: ~neutral per time |

| r16a-d21-b16-lr1e2 | 0.78270 | LR 1e-2 = 8e-3 at d21 (plateau) |
| r16b-d22-b16-normcanon-lr8e3 | 0.78295 | NormCanon saves mem (61.7GB) but SLOWER bwd; 3557 total (nearly timeout!); no gain over d21 |
| r16c-d16-warmup2 | 0.79222 | warmup 2%: -0.0007 |
| r16d-d16-pattern-SL | 0.79218 | pattern SL: -0.0007, +1% time |

| r17a-d21-b16-SL-wu2 | 0.78146 | BEST. SL + warmup 2% at d21. 3255 train / 3406 total |
| r17b-d21-b16-SL-wu2-seed42 | 0.78253 | SEED NOISE ~0.001 (!). Differences < ~0.001 are not reliable |
| r17c-d16-wsd50 | 0.79398 | WSD (decay last 50%) worse +0.0018: keep linear decay from warmup |
| r17d-d16-pattern-L | 0.79138 | all-full attn: -0.0008 vs SSSL+wu2 but +3.4% time |

NOTE: seed noise ~0.001 => at the end, spend leftover budget re-running the best config with several seeds
(score = best of any experiment).

| r18a-d21-bigram | 0.77926 | BEST. hashed bigram emb (65521x1792, zero-init, emb LR): -0.0022. 3370 total, 75GB @ b16 |
| r18b-d23w1664 | 0.78430 | deeper/narrower worse by 0.003. keep d21w1792 |
| r18c-d16-bigram | 0.78987 | bigram at d16: -0.0024 (real) |
| r18d-d16-backout | 0.79164 | backout -0.0006 (noise-level) |

Patch scripts (in /tmp, recreate from experiments/ if lost): bigram.py, trigram.py, chunkce.py (SoftcapCE chunked
lm_head+CE recompute), fusedmlp.py (ReLU2Proj), normcanon.py (slower; avoid), patch_pe.py, docmask.py.

| r19a-d21-bi-tri-chunkce | 0.77900 | trigram ~no gain at d21; chunked CE did not lower peak (76.9GB) |
| r19b-d21-bigram-backout | 0.77822 | BEST. backout -0.0010 at d21 (consistent with d16). 3359 total |
| r19c-d16-bigram262k | 0.78854 | 4x bigram table: -0.0013 vs 65k |
| r19d-d16-bi-tri | 0.78925 | +trigram 65k: -0.0006 (less than bigger bigram) |

| r20a-d21-bigram262k-backout-b8 | 0.77679 | BEST but 3432 train / 3568 TOTAL (too close to 3600!). 470M-param table adds allreduce/Adam time |
| r20b-d21-bigram262k-f768-b16 | 0.78067 | factorized table (emb 768 + Linear up): BAD (+0.0025 vs r19b) |
| r20c-d16-bigram1m-f256 | 0.79228 | factorized 1M x 256: as bad as no bigram. factorization dead |
| r20d-d16-bigram262k-lr3x | 0.79071 | 3x LR on table: +0.0022 worse |

| r21a/r21b (d21 + bf16 flat allreduce) | crash | OOM: flat bf16 buffer kept alive into next step (+2.4GB). fixed with del |
| r21c-d16-bigram262k-lr033 | 0.78761 | table LR/3: -0.0009 |
| r21d-d16-bigram262k-embwd0 | 0.78697 | WD 0 on embedding tables (wte, VE, bigram): -0.0016 |

Names: only letters, digits, -, _ (no dots!).

| r22a-d21-bigram-embwd0-bf16ar | 0.77694 | BEST SAFE (3334 total). emb WD0 -0.0013 at d21; bf16 allreduce 34->19 ms |
| r22b-d21-bigram131k-wd0-lr033-bf16ar | 0.77756 | 131k + table LR/3: no gain over r22a |
| r22c-d16-bigram262k-wd0-lr033 | 0.78719 | WD0 + LR/3: no gain over WD0 alone |
| r22d-d16-bigram262k-wd0-emblr2p5 | 0.78781 | emb mult 2.5: worse. keep 5 |

| r23a-d21-bigram262k-wd0-bf16ar-b8 | 0.77532 | BEST but 3401 train / 3534 total (66s margin: unsafe to grow). step0 compile 79s |
| r23b-d16-headlr1 | 0.78646 | head LR x1.0: -0.0005 (noise) |
| r23c-d16-beta2-099 | 0.78663 | beta2 0.99: -0.0003 (noise) |
| r23d-d16-muonmu098 | 0.79262 | Muon mu 0.98: +0.0057 WORSE |

Overheads at d21: step-0 compile ~80s (d16: 62s), steps 1-10 ~9s, eval+compile ~40s.

| r24a-d21-bigram262k-b16-memsave | 0.77564 | b16 via bf16 Muon momentum + per-step flat buffer: 3384 total (-150s vs r23a) but PEAK 78.96GB (limit 79.17!) |
| r24b-d21-perblock-compile | timeout | per-block compile: compile 42s (vs 81) but fwd/bwd ~12% slower (no cross-block fusion). dead |
| r24c-d16-muonmu09 | 0.78558 | Muon mu 0.9: -0.0014 (vs 0.95) |
| r24d-d16-docmask | 0.78665 | doc masking at d16: -0.0003 (noise) for +1.8% time. dead |

| r25a-d21-262k-b16-mu09 | 0.77503 | BEST. mu 0.9 at d21: -0.0006 vs r24a. 3349 total, 79.0GB peak |
| r25b-d21-262k-b16-mu09-head1-b2 | 0.77523 | + head LR 1 + beta2 0.99: no extra gain |
| r25c-d16-muonmu085 | 0.78639 | mu 0.85 worse than 0.9 (0.78558). keep 0.9 |
| r25d-d16-bigram-ve-last | 0.78482 | last-layer VE indexed by bigram hash (65521): -0.0008 |

| r26a-d21-adambf16-bigramve | 0.77588 | bigram VE on last layer does NOT transfer to d21 (+0.0009) |
| r26b-d21-adambf16 | 0.77482 | BEST. AdamW bf16 moments: same/better quality, peak 76.4GB (-2.6GB); eager step 28ms (slow) |
| r26c-d16-bigram-ve-last2 | 0.78515 | 2 bigram VEs: no extra |
| r26d-d16-mu09-lr12e3 | 0.78423 | mu 0.9 + LR 1.2e-2: -0.0013 vs LR 1e-2 |

| r27a-d21-adambf16c-lr1e2 | 0.77333 | BEST (tie). LR 1e-2 with mu 0.9: -0.0015. compiled AdamWBF16 8.4ms. 3371 total, 76.4GB |
| r27b-d21-adambf16c-lr12e3 | 0.77334 | LR 1.2e-2: same |
| r27c-d16-mu09-lr15e3 | 0.78469 | LR 1.5e-2 worse than 1.2e-2 at d16 |
| r27d-d16-bigram-out | 0.78315 | output-side bigram table (65521, before final norm): -0.0011 |

| r28a-d21-bigramout-lr1e2 | 0.77278 | bigram_out at d21: -0.0006 |
| r28b-d21-bigramout-lr12e3 | 0.77269 | BEST. 3380 total, peak 78.0GB |
| r28c-d16-bigramout262k | 0.78322 | bigger output table: no gain |
| r28d-d16-trigramout | 0.78244 | + output trigram table: -0.0007 |

CAUTION: removing a param from forward -> grad None -> crashes flat allreduce/AdamWBF16. Use buffers.
(r29a-d were cancelled by me before running -> renamed r29e-h.)

| r29e-d21-trigramout-noresid | crash | OOM (no memory saved by removing resid_lambdas) |
| r29f-d21-noresid | 0.77469 | no resid_lambdas: +0.002 worse, same peak memory. dead |
| r29g-d16-noresid | 0.78549 | +0.003 worse at d16 |
| r29h-d16-midbigram | 0.78531 | mid-layer bigram: nothing (vs r29g) |

d16 proxy now = r28d (0.78244): d16 + bigram262k + emb WD0 + mu0.9 + LR1.2e-2 + bigram_out + trigram_out.

| r30a-d21-r28b-seed42 | 0.77256 | BEST. seed 42 vs 1337: 0.00013 apart (noise smaller here) |
| r30b-d16-softcap20 | 0.78260 | softcap 20: neutral |
| r30c-d16-attnscale014 | 0.78244 | attn scale 0.14: identical |
| r30d-d16-nosmear | 0.78214 | no smear: -0.0003 (neutral) -> new d16 proxy |

| r31a-d21-nosmear | crash | OOM at step 0 inside flat all_reduce (torch.cat fp32 temp 6.4GB). -> /tmp/bf16ar2.py (cast into prealloc bf16 buffer) |
| r31b-d16-win512 | 0.78222 | window 512: neutral |
| r31c-d16-wavg | 0.78277 | weight averaging last 150 steps: +0.0006 worse |
| r31d-d16-heads64 | 0.78460 | 28 heads x 64: +0.0024 worse |

Ideas from modded-nanogpt records (fetched README): bigram sign trick (#83), learnable XSA (#82: y -= tanh(a_h)<y,v^>v^),
MUDD skip connections (#81), prefix-token-prediction aux loss (#88), higher min LR + EMA (#72), paired-head Q/K
orthogonalization in Muon (#80), hashed n-gram 84.6M rows sharded (#92), sparse bigram grad comms (#71).

| r32a-d21-nosmear-ar2 | crash | NCCL OOM in first all_reduce again (PyTorch cache holds memory) -> /tmp/emptycache.py (empty_cache at step 0) |
| r32b-d21-bigram196k-trigramout | crash | OOM in backward: trade does not fit |
| r32c-d16-signtrick | 0.78154 | bigram sign trick (/tmp/signtrick.py): -0.0006 |
| r32d-d16-xsa | crash | OOM (fp32 XSA intermediates) |

MEMORY is now the binding constraint at d21 (~78GB peak of 79.2) and close at d16 (~72GB).

| r33a-d21-nosmear-sign | crash | OOM in backward (no-smear variant oddly heavier). keep smear |
| r33b-d21-smear-sign | 0.77206 | BEST. sign trick at d21: -0.0005. 3412 total, 78.0GB |
| r33c-d16-xsa-bf16 | crash | OOM (old d16 proxy is at 77GB!) |
| r33d-d16-sign-outtables | 0.78110 | sign on output tables: -0.0004 |

| r34a-d21-sign-out | 0.77176 | BEST. sign on bigram_out: -0.0003. 3435 total, 78.1GB |
| r34b-d16proxy | 0.78096 | NEW d16 proxy (= r34a @16): 61.6GB, 2469s |
| r34c-d16proxy-xsa | 0.77773 | learnable XSA (/tmp/xsa.py): -0.0032 !! (+4% time, +2.7GB) |
| r34d-d16proxy-minlr005 | 0.78207 | LR floor 5%: +0.0011 worse |

| r35a-d20-xsa | 0.77082 | BEST. d20+XSA beats d21 no-XSA. 3340 total, 78.1GB (seed 42 lineage) |
| r35b-d19-xsa | 0.77152 | 3207 total, 74.9GB |
| r35c-d16proxy-xsa-fullonly | 0.77896 | XSA on L layers only: -0.0020 (vs -0.0032 all layers) |
| r35d-d16proxy-xsa-trigramout | 0.77668 | + output trigram (sign, /tmp/trigramout_sign.py): -0.0010 -> new d16 proxy |

| r36a-d19-xsa-trigramout | 0.77082 | d19 XSA + trigram_out = d20 XSA. 3260 total, 76.7GB |
| r36b-d20-xsa-seed1337 | 0.77062 | BEST (seed draw of r35a). 3361 total |
| r36c-d16-xsa-init05 | 0.77658 | XSA gate init 0.5: neutral |
| r36d-d16-xsa-trigramin | 0.77710 | input trigram (sign): +0.0004, no |

Prefix-token-prediction aux loss (#88) is MTP-like; MTP hurt us -> skipped.

| r37a-d19-xsa-trigram-seed1337 | 0.77099 | seed draw: ~same |
| r37b-d16-mlp5x | 0.77280 | MLP 5x: -0.0039 !! (+10% time 2846s, +7.5GB -> 69GB) |
| r37c-d16-gatedskips | 0.77541 | gated U-net skips (2*sigmoid(Linear(32,1))): -0.0013 (/tmp/gatedskips.py) |
| r37d-d16-cautiouswd | 0.77806 | cautious WD: +0.0014 worse |

| r38a-d17-mlp5x-gskip | 0.77016 | 3188 total, 73GB |
| r38b-d18-mlp5x-gskip | 0.76900 | BEST. d18 MLP5x gated skips XSA trigram_out. 3359 total, 76.6GB |
| r38c-d16-mlp6x | 0.77061 | MLP 6x at d16 (no gskip): -0.0022 vs 5x; 3250 total, 72GB |
| r38d-d16-mlp5x-gskip | 0.77205 | new d16 proxy (MLP5x + gskip) |

MLP width >> depth per unit time now. Block FLOP units: L*(4+2m) d^2.

| r39a-d16-mlp6x-gskip | 0.76913 | 3255 total, 72.4GB -> new d16 proxy |
| r39b-d17-mlp6x-gskip | 0.76775 | BEST. 3307 train / 3456 total (tight), 76.2GB |
| r39c-d14-mlp7x-gskip | 0.77077 | 3107 total |
| r39d-d13-mlp8x-gskip | 0.77145 | 3090 total. too shallow |

| r40a-d17-mlp6x-seed42 | 0.76736 | BEST. 3470 total (130s margin), 76.2GB |
| r40b-d16m6-lr1e2 | 0.76982 | LR 1e-2 worse; keep 1.2e-2 |
| r40c-d16m6-lr14e3 | 0.76974 | LR 1.4e-2 worse |
| r40d-d16m6-nove | 0.77213 | no value embeddings: +0.003 for -2% time. keep VE |

| r41a-d17m6-signbf16 | 0.76756 | NOT faster: 3368 train / 3539 total (!! node variance?). d17m6 too close to limit |
| r41b-d16m6-x0gates | 0.76935 | dynamic x0 gates: neutral |
| r41c-d16m6-bigram393k | 0.76931 | bigger bigram table: neutral now |
| r41d-d16m6-attngate | 0.76795 | head-wise attn output gate 2*sig(W x[:32]): -0.0012, but +3.6GB (76GB) |

| r42a-d16m6-gate-fused | 0.76767 | SAFE BEST (3278 total). fused XSA+gate: same loss, but NO memory saving (76GB) |
| r42b-d17m6-gate-fused | crash | OOM |
| r42c-d16m6-gate-mlpgate | 0.77047 | per-token MLP gate: worse |
| r42d-d16m6-gate-fullx | 0.77033 | gate from full x: worse |

| r43a-d16m6-gate-seed42 | 0.76831 | seed draw (+0.0006) |
| r43b-d16m6-gate-memdiag | 0.76772 | replicate of r42a (0.76767). MEMORY: static 9.7G (no grads) / 15.9G (with grads); micro0 fwd -> 66.3G, micro1 fwd -> 72.5G (fp32 grads of micro0 resident!), peak 74.2G in micro1 bwd |
| r43c-d16m6-elemgate | 0.76794 | elementwise gate = head-wise, costlier |
| r43d-d16m65-gate | 0.76693 | BEST. d16 MLP 6.5x + gate. 3446 total, 77.5GB |

| r44a-d16m65-gate-nocanon | crash | IndentationError (my edit) -> retried as r45c |
| r44b-d16m65-gate-stash | 0.76622 | grad stash: 74.2GB (-3.3GB), quality same/better |
| r44c-d16m65-gate-canon2 | 0.76593 | BEST. canon kernel 2: better, faster (3160 train/3294 total), 70.3GB (-7GB!) |
| r44d-d16m65-gate-seed42 | 0.76688 | seed draw (same) |

| r45a-d16m7-canon2-stash | 0.76496 | BEST. d16 MLP7x canon2 stash gate. 3320 train / 3456 total, 68.4GB |
| r45b-d17m6-canon2-stash | 0.76549 | 3410 total, 69GB |
| r45c-d16m65-gate-nocanon | 0.77028 | no canon: +0.0035 -> canon matters |
| r45d-d16m65-canon2-stash | 0.76588 | new proxy (67GB, 3333 total) |

TIME binds now (memory ~68GB). 

| r46a-d16m65-canon3 | 0.76612 | canon k3: neutral/worse than k2 |
| r46b-d16m65-veall | 0.76563 | VE all layers: neutral (-0.0002) |
| r46c-d16m7-seed42 | 0.76493 | BEST (tie with r45a 0.76496). 3465 total |
| r46d-d16m65-win128 | 0.76621 | window 128: neutral, not faster |

| r47a-d16m65-fourgramout | 0.76533 | 4-gram output table (sign): -0.0006 (/tmp/fourgram.py) |
| r47b-d16m65-warmup1 | 0.76799 | warmup 1%: +0.0021 worse. keep 2% |
| r47c-d16m65-gqa7 | 0.76813 | GQA 7kv: +0.0022 but -7% time, -9.5GB. net loss |
| r47d-d16m65-emb7 | 0.76552 | emb LR mult 7: -0.0004 |

| r48a-d16m7-4g-emb7-s42 | 0.76461 | BEST. 3499 total (100s margin!) |
| r48b-d16m7-4g-emb7-s1337 | 0.76479 | other seed |
| r48c-d16m65-emb9 | 0.76529 | emb LR 9: -0.0002 vs 7 |
| r48d-d16m65-muonlr12 | 0.76640 | Muon LR x1.2 (vs AdamW): +0.0005 worse |

| r49a-d16m7-nosync | 0.76442 | BEST. no per-step syncs: ~same time (GPU-bound), 3470 total |
| r49b-d16m7-emb9 | 0.76461 | emb 9 at d16m7: neutral |
| r49c-d16m65-muonlr085 | 0.76547 | Muon x0.85 vs AdamW: -0.0004 |
| r49d-d16m65-nosmear | 0.76560 | no smear: -0.0003 |

## Round 50 (done)
- r50a/b r49a + Muon x0.85 + no smear (seeds 42/1337), r50c proxy QK-norm gains, r50d proxy final-norm gain.
| r50a-d16m7-mu085-nosmear-s42 | 0.76431 | 3538 total (!) |
| r50b-d16m7-mu085-nosmear-s1337 | 0.76396 | BEST. Muon x0.85 + no smear. 3496 total (too close) |
| r50c-d16m65-qkgain | 0.76660 | QK-norm gains: +0.0007 worse |
| r50d-d16m65-finalgain | 0.76632 | final norm gain: +0.0004 worse |

## Round 51 (done)
- r51a/b r50b with MLP hidden 12288 (time margin), seeds 1337/42; r51c proxy emb beta1 0.8; r51d proxy grad clip 2x EMA.
| r51a-d16h12288-s1337 | 0.76446 | hidden 12288: same quality, 3449 total (safer) |
| r51b-d16h12288-s42 | 0.76423 | 3420 total. NEW BASE for safety |
| r51c-d16m65-embbeta08 | 0.76599 | emb beta1 0.8: neutral |
| r51d-d16m65-gradclip | 0.76566 | grad clip 2x EMA: neutral |

## Round 52 (done)
- shape sweep at ~equal FLOPs from r51b: w1536 d19, w2048 d12, w1664 d18, w1920 d14 (MLP ~7x).
| r52a-w1536d19 | 0.76673 | 3228 total. worse |
| r52b-w2048d12 | timeout | |
| r52c-w1664d18 | timeout (0.76452) | printed val at 3582 total but harness killed it: EFFECTIVE LIMIT ~3575s total_seconds |
| r52d-w1920d14 | 0.76459 | 3465 total. ~= w1792 d16 |
=> w1792 d16 h12288 stays.

## Round 53 (done)
- r53a proxy LR decay power 1.3, r53b power 0.75, r53c proxy skip-bigram output table, r53d r51b seed 7.
| r53a-proxy-decay13 | 0.76690 | decay power 1.3: worse |
| r53b-proxy-decay075 | 0.76760 | decay power 0.75: worse. linear stays |
| r53c-proxy-skipbigram | 0.76593 | skip-bigram out table: neutral |
| r53d-d16h12288-s7 | 0.76444 | seed draw |

## Round 54 (done)  MoE test (/tmp/moe/moe.py, /tmp/moepatch.py)
- r54a proxy MoE 2 experts x 10752, cf 1.1 vs r54b dense 6x control; r54d MoE cf 1.25; r54c r51b seed 123.
| r54a-proxy-moe2-h10752 | crash | MoE OOM (no relu^2 recompute + 616M extra params) |
| r54b-proxy-dense-m6 | 0.76734 | dense 6x control (3208 total, 65.6GB) |
| r54c-d16h12288-s123 | 0.76460 | seed draw |
| r54d-proxy-moe2-cf125 | crash | OOM |

## Round 55 (done) MoE with batched ReLU2BMM recompute (/tmp/moe/moe2.py, /tmp/moepatch2.py)
- r55a MoE 2x10752 cf1.1, r55b MoE 4x5376 cf1.1, r55d MoE 2x10752 cf1.0; r55c r51b seed 2024.
| r55a/r55d (MoE 2x10752) | crash | OOM again. |
| r55b-proxy-moe4-h5376 | 0.77864 | MoE 4x3x: much worse than dense 6x (0.76734) even per time. MoE DEAD |
| r55c-d16h12288-s2024 | 0.76514 | seed draw. r51b seeds: 42 .76423, 1337 .76446, 7 .76444, 123 .76460, 2024 .76514 (std ~0.0003) |

## Round 56 (done)
- r56a proxy+4g + 5-gram out table, r56b proxy BOS-target masking, r56c proxy head LR 0.6, r56d r50b seed 7.
| r56a-proxy-fivegramout | 0.76540 | 5-gram: neutral (vs 4g 0.76533) |
| r56b-proxy-bosmask | 0.76551 | BOS targets masked in train loss: -0.0004 (/tmp/bosmask.py) |
| r56c-proxy-headlr06 | 0.76581 | head LR 0.6: neutral |
| r56d-r50b-seed7 | 0.76414 | seed draw of best (3493 total) |

## Round 57 (done)
- r57a proxy 7 heads x256, r57b proxy NoPE on full layers, r57c r50b+bosmask, r57d r51b+bosmask.
| r57a-proxy-hd256 | 0.76784 | 7x256 heads: +0.0019 worse |
| r57b-proxy-nope-full | 0.76829 | NoPE on full layers: +0.0024 worse |
| r57c-r50b-bosmask | 0.76398 | bosmask at scale: no gain |
| r57d-r51b-bosmask | 0.76445 | no gain |

## Round 58 (done) proxy = r49c (Muon x0.85, 0.76547)
- r58a Muon x0.7, r58b WD 0.15, r58c WD 0.07, r58d softcap 12.
| r58a-proxy-muon07 | 0.76541 | Muon x0.7: neutral vs x0.85 |
| r58b-proxy-wd015 | 0.76654 | WD 0.15 worse |
| r58c-proxy-wd007 | 0.76589 | WD 0.07 worse. keep 0.1 |
| r58d-proxy-softcap12 | 0.76596 | softcap 12 worse. keep 15 |

## Round 59 (done)
- r59a proxy bf16 residual stream (/tmp/bf16resid.py), r59b r50b bf16 residual, r59c r50b seed 99, r59d r51b seed 31337.
| r59a-proxy-bf16resid | 0.76505 | bf16 residual: -0.0004 and -86s train (-2.7%) |
| r59b-r50b-bf16resid | 0.76419 | bf16 residual at r50b: ~same quality, 3253 train / 3392 total (-100s). NEW MAIN |
| r59c-r50b-seed99 | TIMEOUT | r50b (3500s expected) timed out on a slow node: node variance up to +100s! |
| r59d-r51b-seed31337 | 0.76441 | seed draw (3466 total vs usual 3420) |
RULE: keep expected total <= ~3450s.

## Round 60 (done)
- r60a r59b + hidden 12800, r60b r59b + bosmask, r60c/d r59b seeds 42/7.
| r60a-bf16-h12800 | 0.76382 | BEST. bf16 residual + hidden 12800. 3280 train / 3414 total |
| r60b-bf16-bosmask | 0.76397 | |
| r60c-bf16-s42 | 0.76403 | r59b seeds: 1337 .76419, 42 .76403, 7 .76407 |
| r60d-bf16-s7 | 0.76407 | |

## Round 61 (done)
- r61a/b r60a seeds 42/7, r61c r60a + bosmask, r61d proxy Adam eps 1e-8.
| r61a-h12800-s42 | 0.76342 | BEST (seed 42 of r60a config). 3412 total |
| r61b-h12800-s7 | 0.76371 | |
| r61c-h12800-bosmask | 0.76390 | bosmask: neutral at scale |
| r61d-proxy-adameps1e8 | timeout | (proxy normally 3230s -> stuck/slow node) |

## Round 62 (done)
- r62a/b r60a config seeds 123/2024, r62c proxy Muon attn LR x1.25, r62d x0.8.
| r62a-h12800-s123 | 0.76372 | seeds of r60a config: 1337 .76382, 7 .76371, 42 .76342, 123 .76372, 2024 .76376 (std .00015) |
| r62b-h12800-s2024 | 0.76376 | |
| r62c-proxy-attnlr125 | 0.76506 | Muon attn LR x1.25: neutral |
| r62d-proxy-attnlr08 | 0.76605 | x0.8: worse |

## Round 63 (done)
- r63a proxy value residual (scalar), r63b value residual (gated) (/tmp/vres.py), r63c/d r60a seeds 8/9.
| r63a-proxy-vres-scalar | 0.76558 | value residual (scalar): +0.0005 worse (not in results.tsv: harness crashed on disk quota) |
| r63b-proxy-vres-gate | 0.76544 | value residual gated: worse |
| r63c-h12800-s8 | 0.76412 | seed draw |
| r63d-h12800-s9 | no result | |
DISK QUOTA outage (~5h) after round 63: scratch is 97% full; keep outputs small.

## Round 64 (done)
- r64a proxy last-3 attn read x_backout (/tmp/backin.py), r64b proxy v scale, r64c proxy x01 table, r64d best cfg seed 11.
| r64a-proxy-backin3 | 0.76465 | last 3 attn layers read x_backout (PR #241): -0.0004 |
| r64b-proxy-vscale | 0.76513 | v scale: neutral |
| r64c-proxy-x01 | 0.76547 | extra embedding table: worse |
| r64d-h12800-s11 | 0.76394 | seed draw |

## Round 65 (done)
- r65a/b best config + backin3 (seeds 42/1337), r65c proxy backin5, r65d proxy backin2.
| r65a-best-backin3-s42 | 0.76230 | backin3 at scale: -0.0011 |
| r65b-best-backin3-s1337 | 0.76202 | BEST. -0.0018 vs r60a same seed |
| r65c-proxy-backin5 | 0.76423 | k=5 better than 3 at proxy |
| r65d-proxy-backin2 | 0.76430 | |

## Round 66 (done)
- r66a/b best + backin5 (seeds 1337/42), r66c proxy backin7, r66d proxy backin5 with backout at L/2.
| r66a-best-backin5-s1337 | 0.76191 | BEST. backin5 at scale |
| r66b-best-backin5-s42 | 0.76255 | |
| r66c-proxy-backin7 | crash | k=7 reads x_backout before it is set (layer 9). bug |
| r66d-proxy-backin5-bo-half | 0.76457 | cache at L/2: worse than 2L/3 |

## Round 67 (done)
- r67a proxy backin5 + learnable mix, r67b proxy backin6, r67c best seed 7, r67d proxy cache after layer 10 + backin5.
| r67a-proxy-backin5-mix | 0.76388 | learnable mix (init 0) of cached/current for attn input: -0.0004 (/tmp/backinmix.py) |
| r67b-proxy-backin6 | 0.76466 | k=6 worse; keep 5 |
| r67c-best-backin5-s7 | 0.76239 | seed draw |
| r67d-proxy-backin5-bo10 | 0.76454 | cache after layer 10: worse; keep 2L/3-1 |

## Round 68 (done)
- r68a/b best + backin mix (seeds 1337/42), r68c proxy dynamic mix gate, r68d proxy MLP mix too.
| r68a-best-backinmix-s1337 | 0.76187 | BEST. backin5 + learnable mix (/tmp/backinmix.py) |
| r68b-best-backinmix-s42 | 0.76189 | |
| r68c-proxy-mixgate | 0.76529 | dynamic mix gate: worse |
| r68d-proxy-mlpmix | 0.76401 | MLP mix too: neutral |

## Round 69 (done)
- r69a/b best (r68a) seeds 7/2024, r69c proxy second cache (layer 3 -> attn 4-8), r69d proxy no final backout subtraction.
| r69a-best-mix-s7 | 0.76230 | seeds of r68a config: 1337 .76187, 42 .76189, 7 .76230, 2024 .76240 |
| r69b-best-mix-s2024 | 0.76240 | |
| r69c-proxy-cache2 | 0.76419 | second cache for middle layers: slightly worse |
| r69d-proxy-nobackoutsub | 0.76404 | no final backout subtraction: neutral |

## Round 70 (done)
- r70a proxy attention lane, r70b proxy early layers attn read x0 mix, r70c/d best seeds 123/99.
| r70a-proxy-attnlane | 0.76400 | attention lane (x_backout + attn outputs): neutral |
| r70b-proxy-early-x0in | 0.76523 | early attn reads x0 mix: worse |
| r70c-best-mix-s123 | 0.76191 | seed draw |
| r70d-best-mix-s99 | 0.76208 | seed draw |

## Round 71 (done)
- r71a proxy logit bias, r71b proxy per-layer attention temperature, r71c/d best seeds 5/31337.
| r71a-proxy-logitbias | 0.76384 | logit bias: neutral |
| r71b-proxy-attntemp | 0.76465 | per-layer attention temperature: worse |
| r71c-best-mix-s5 | 0.76233 | seeds r68a config (8): mean .76216, sd .0002, best .76187 |
| r71d-best-mix-s31337 | 0.76246 | |

## Round 72 (done)
- r72a proxy pattern SLL, r72b proxy MLP width ramp up with depth, r72c ramp down, r72d best seed 77.
| r72a-proxy-patternSLL | 0.76401 | SLL: neutral |
| r72b-proxy-mlpramp-up | 0.76499 | MLP width ramp up: worse |
| r72c-proxy-mlpramp-down | 0.76484 | ramp down: worse |
| r72d-best-mix-s77 | 0.76219 | seed |

## Round 73 (done)
- r73a proxy per-channel backout lambda, r73b proxy no XSA in backin layers, r73c/d best seeds 88/66.
| r73a-proxy-backoutvec | 0.76385 | per-channel backout: neutral |
| r73b-proxy-noxsa-backin | 0.76480 | no XSA in backin layers: worse |
| r73c-best-mix-s88 | 0.76307 | seed (wider spread than thought) |
| r73d-best-mix-s66 | 0.76290 | seed |

## Round 74 (done)
- r74a proxy Muon momentum warmup, r74b proxy head LR 0.25, r74c proxy emb mult 7, r74d best seed 21.
| r74a-proxy-muwarm | 0.76369 | Muon mu warmup 0.8->0.9 (5 levels): -0.0002 (/tmp/muwarm.py) |
| r74b-proxy-headlr025 | 0.76489 | head LR 0.25: worse |
| r74c-proxy-emb7 | 0.76380 | emb 7: ~neutral |
| r74d-best-mix-s21 | 0.76250 | seed |

## Round 75 (done)
- r75a/b best + mu warmup (seeds 1337/42), r75c/d proxy MLP Muon LR x1.2 / x0.8.
| r75a-best-muwarm-s1337 | 0.76149 | mu warmup at scale: -0.0004 vs same seed |
| r75b-best-muwarm-s42 | 0.76123 | BEST. -0.0007 vs same seed |
| r75c-proxy-mlplr12 | 0.76458 | MLP Muon LR x1.2: worse |
| r75d-proxy-mlplr08 | 0.76418 | x0.8: worse |

## Round 76 (done)
- muwarm variants at scale (seed 42): r76a start 0.7, r76b over 600 steps, r76c + cooldown to 0.85 in last 500; r76d seed 7.
| r76a-muwarm-from07 | 0.76278 | warmup from 0.7: worse |
| r76b-muwarm-600 | 0.76127 | 600-step warmup: same |
| r76c-muwarm-cooldown | 0.76175 | mu cooldown at end: worse |
| r76d-muwarm-s7 | 0.76157 | seed 7 (vs .76230 without warmup) |

## Round 77 (done)
- r77a warmup 0.85->0.9, r77b warmup 0.8->0.92, r77c/d muwarm seeds 2024/123.
| r77a-muwarm-from085 | 0.76096 | warmup 0.85->0.9: better than 0.8->0.9 |
| r77b-muwarm-to092 | 0.76068 | BEST. warmup 0.8->0.92 |
| r77c-muwarm-s2024 | 0.76176 | seed |
| r77d-muwarm-s123 | 0.76120 | seed |

## Round 78 (done)
- momentum warmup grid (seed 42): 0.85->0.92, 0.8->0.95, 0.85->0.95, 0.8->0.94.
| r78a-muwarm-085-092 | 0.76076 | grid: all within ~0.0006 (0.8->0.9 .76123 ... 0.8->0.92 .76068); keep 0.8->0.92 |
| r78b-muwarm-08-095 | 0.76098 | |
| r78c-muwarm-085-095 | 0.76074 | |
| r78d-muwarm-08-094 | 0.76132 | |

## Round 79 (done)
- r79a best + LR warmup 3%, r79b best + Adam beta1 warmup, r79c/d best seeds 1337/7.
| r79a-best-lrwarm3 | 0.76145 | LR warmup 3%: worse |
| r79b-best-b1warm | 0.76129 | Adam beta1 warmup: worse |
| r79c-best-s1337 | 0.76148 | seed |
| r79d-best-s7 | 0.76116 | seed |

## Round 80 (done)
- r80a mu rises to 0.96 late, r80b Muon RMS 0.19, r80c/d best seeds 2024/123.
| r80a-best-mu-late096 | 0.76143 | momentum rising late: worse |
| r80b-best-muonrms019 | 0.76104 | Muon RMS 0.19: ~neutral |
| r80c-best-s2024 | 0.76137 | r77b seeds: 42 .76068, 1337 .76148, 7 .76116, 2024 .76137, 123 .76106 (mean .7611) |
| r80d-best-s123 | 0.76106 | |

## Round 81 (done)
- r81a best + attention lane (at scale), r81b best backin k=4, r81c/d best seeds 99/5.
| r81a-best-attnlane | 0.76140 | attention lane at scale: no gain |
| r81b-best-backin4 | 0.76087 | k=4 ~ k=5 (same seed .76068) |
| r81c-best-s99 | 0.76132 | seed |
| r81d-best-s5 | 0.76166 | seed. r77b config 7 seeds mean ~.7612 |

## Round 82 (done)
- r82a best MLP mu target 0.95, r82b attn mu target 0.95, r82c/d best seeds 31337/77.
| r82a-best-mu-mlp095 | 0.76126 | MLP momentum target 0.95: worse |
| r82b-best-mu-attn095 | 0.76054 | BEST. attention momentum target 0.95 (p.mu_mult) |
| r82c-best-s31337 | 0.76103 | seed |
| r82d-best-s77 | 0.76142 | seed |

## Round 83 (done)
- r83a attn mu 0.97, r83b attn 0.95 + MLP 0.90, r83c/d r82b seeds 1337/7.
| r83a-mu-attn097 | 0.76038 | BEST. attention momentum target 0.97 |
| r83b-mu-attn095-mlp090 | 0.76070 | MLP 0.90: neutral/worse |
| r83c-attn095-s1337 | 0.76069 | r82b seeds: 42 .76054, 1337 .76069, 7 .76066 (robust, mean -0.0006 vs r77b) |
| r83d-attn095-s7 | 0.76066 | |

## Round 84 (done)
- r84a attn mu 0.98, r84b 0.99, r84c qk 0.98 / vo 0.95, r84d r83a seed 1337.
| r84a-mu-attn098 | 0.76069 | attn mu 0.98: slightly worse |
| r84b-mu-attn099 | 0.76172 | 0.99: worse. optimum ~0.97 |
| r84c-mu-qk098-vo095 | 0.76090 | qk/vo split: worse |
| r84d-attn097-s1337 | 0.76052 | seed |

## Round 85 (done)
- r85a emb Adam beta1 0.95, r85b 0.85, r85c/d r83a seeds 7/2024.
| r85a-emb-beta095 | 0.76066 | emb beta1 0.95: worse |
| r85b-emb-beta085 | 0.76036 | BEST (tiny). emb beta1 0.85 |
| r85c-attn097-s7 | 0.76078 | r83a seeds: 42 .76038, 1337 .76052, 7 .76078, 2024 .76117 |
| r85d-attn097-s2024 | 0.76117 | |

## Round 86 (done)
- r86a emb beta1 0.8, r86b all AdamW beta1 0.85, r86c/d r85b seeds 1337/7.
| r86a-emb-beta08 | 0.76052 | emb beta1 0.8: ~same/worse than 0.85 |
| r86b-all-beta085 | 0.76093 | all AdamW beta1 0.85: worse |
| r86c-embb085-s1337 | 0.76040 | r85b seeds: 42 .76036, 1337 .76040, 7 .76065 |
| r86d-embb085-s7 | 0.76065 | |

## Round 87 (done)
- r87a attention mu 0.97 only layers 8-15, r87b only layers 0-7, r87c/d r85b seeds 2024/123.
| r87a-attnmu-late | 0.76071 | attn mu 0.97 only layers 8-15: worse than all layers |
| r87b-attnmu-early | 0.76059 | only layers 0-7: worse than all |
| r87c-embb085-s2024 | 0.76108 | r85b seeds mean (5) ~.7607 |
| r87d-embb085-s123 | 0.76078 | (3501s total: close to limit!) |

## Round 88 (done)
- r88a Nesterov off, r88b emb beta2 0.99, r88c/d r85b seeds 99/5.
| r88a-best-nonesterov | 0.76025 | BEST (tiny). Muon Nesterov off |
| r88b-best-embbeta2-099 | 0.76065 | emb beta2 0.99: worse |
| r88c-embb085-s99 | 0.76105 | seed (3503s total!) |
| r88d-embb085-s5 | 0.76097 | seed |

## Round 89 (done)
- r89a/d r88a seeds 1337/7, r89b no-Nesterov attn mu 0.95, r89c no-Nesterov global mu 0.9.
| r89a-nonest-s1337 | 0.76045 | seed |
| r89b-nonest-attn095 | 0.76007 | no-Nesterov attn 0.95: better than 0.97 |
| r89c-nonest-mu09 | 0.75990 | BEST. no Nesterov + global mu target 0.9 (warmup 0.8->0.9), attn x0.97/0.92 |
| r89d-nonest-s7 | 0.76047 | seed |

## Round 90 (done)
- r90a global mu 0.88, r90b global 0.9 + attn ratio 0.95/0.92, r90c/d r89c seeds 1337/7.
| r90a-nonest-mu088 | 0.76005 | global 0.88: worse |
| r90b-nonest-mu09-attn095r | 0.75984 | BEST. global 0.9 + attn ratio 0.95/0.92 |
| r90c-mu09-s1337 | 0.75987 | r89c seeds: 42 .75990, 1337 .75987, 7 .76001 (mean .7599) |
| r90d-mu09-s7 | 0.76001 | |

## Round 91 (done)
- r91a MLP mu 0.88, r91b warmup from 0.85, r91c/d r90b seeds 1337/7.
| r91a-mlpmu088 | 0.76011 | MLP mu 0.88: worse |
| r91b-muwarm-from085 | 0.76038 | warmup from 0.85: worse (keep 0.8) |
| r91c-r90b-s1337 | 0.75982 | BEST (seed). r90b seeds: 42 .75984, 1337 .75982, 7 .76019 |
| r91d-r90b-s7 | 0.76019 | |

## Round 92 (done)
- r92a MLP mu 0.92, r92b warmup 450 steps, r92c/d r90b seeds 123/2024.
| r92a-mlpmu092 | 0.76030 | MLP mu 0.92: worse |
| r92b-muwarm450 | 0.75999 | warmup 450 steps: same |
| r92c-r90b-s123 | 0.76009 | r90b seeds (5): .75984 .75982 .76019 .76009 .75983 (mean .75995) |
| r92d-r90b-s2024 | 0.75983 | |

## Round 93 (done)
- r93a LR 1.1e-2, r93b LR 1.3e-2, r93c/d r90b seeds 99/5.
| r93a-lr11e3 | 0.76013 | LR 1.1e-2: worse |
| r93b-lr13e3 | 0.76040 | LR 1.3e-2: worse. keep 1.2e-2 |
| r93c-r90b-s99 | 0.76034 | seed |
| r93d-r90b-s5 | 0.75998 | seed. r90b 7 seeds mean ~.7600 |

## Round 94 (done)
- r94a attention ratio 0.96/0.92; r94b-d r90b seeds 31337/77/11.
| r94a-attnr096 | 0.76006 | attn ratio 0.96: no gain |
| r94b-r90b-s31337 | 0.76018 | seeds |
| r94c-r90b-s77 | 0.76015 | |
| r94d-r90b-s11 | 0.76006 | r90b 10 seeds: best .75982, mean ~.76004 |

## Round 95 (done)
- r95a emb LR mult 6; r95b-d r90b seeds 3/17/23.
| r95a-emb6 | 0.75983 | emb mult 6: same |
| r95b-r90b-s3 | 0.76017 | seeds |
| r95c-r90b-s17 | 0.76017 | |
| r95d-r90b-s23 | 0.76017 | |

## Round 96 (done)
- r96a MLP hidden 12928; r96b-d r90b seeds 1001/2002/3003.
| r96a-h12928 | 0.75983 | hidden 12928: same quality, 3505s total (too risky) |
| r96b-r90b-s1001 | 0.76014 | seeds |
| r96c-r90b-s2002 | 0.76026 | |
| r96d-r90b-s3003 | 0.76024 | |

## Round 97 (done)
- r97a backin k=4; r97b-d r90b seeds 4/8/16.
| r97a-backin4 | 0.76007 | k=4: slightly worse |
| r97b-r90b-s4 | 0.759795 | BEST (seed 4 of r90b config) |
| r97c-r90b-s8 | 0.75997 | |
| r97d-r90b-s16 | 0.75994 | |

## Round 98 (done)
- r98a finer momentum warmup (10 levels); r98b-d r90b seeds 32/64/128.
| r98a-muwarm10lvl | 0.76004 | finer warmup: no gain |
| r98b-r90b-s32 | 0.76046 | seeds |
| r98c-r90b-s64 | 0.76000 | |
| r98d-r90b-s128 | 0.75990 | |

## Round 99 (done)
- r99a logit bias at scale; r99b-d r90b seeds 256/512/1024.
| r99a-logitbias | 0.76004 | logit bias at scale: neutral |
| r99b-r90b-s256 | 0.76048 | seeds |
| r99c-r90b-s512 | 0.75994 | |
| r99d-r90b-s1024 | 0.76008 | |

## Round 100 (done)
- r100a attention ratio 0.94; r100b-d r90b seeds 2048/4096/8192.
| r100a-attnr094 | 0.75999 | attn ratio 0.94: no gain |
| r100b-r90b-s2048 | 0.76029 | seeds |
| r100c-r90b-s4096 | 0.76013 | |
| r100d-r90b-s8192 | 0.76032 | |

## Round 101 (final)
- r101a backout lambda init 0.1; r101b-d r90b seeds 6/9/13.
| r101a-backoutinit01 | 0.76006 | backout init 0.1: no gain |
| r101b-r90b-s6 | 0.75981 | seed |
| r101c-r90b-s9 | 0.76018 | seed |
| r101d-r90b-s13 | 0.76043 | seed |

## Round 101 (done). BUDGET EXHAUSTED (400/400).
FINAL BEST: r97b-r90b-s4 = 0.759795 (r90b config, seed 4). Next: r101b 0.75981, r91c 0.75982, r92d 0.75983.
