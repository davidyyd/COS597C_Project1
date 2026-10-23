# r52a with the pointer computed by two block-sparse flex passes (instead of dense T x T chunks)
Parent: r52a-r51b-docmask-full. Type: exploit (implementation / speed).
Same model as r52a (0.929378 @ 265 s). The earlier flex pointer crashed in Inductor's lowering; the
trigger was the score_mod form `score + where(cond, c1, c2)`; `where(cond, score, c)` lowers fine (as in the
working doc mask). CPU check: flex and dense pointer give the same NLL up to bf16 precision (mean 8.4562 vs
8.4569). The flex version skips the T x T materialisation; if it is much cheaper, r52a fits the cap.
Confirm: ≈ 0.9294 at < 261.7 s.
