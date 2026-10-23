This run uses **search over a population of training scripts**, in rounds of {BATCH} experiments trained in
parallel. There is no single current script: every experiment you have run is a possible starting
point, and you choose what to build on. Your score is the best val_bpb of any experiment.

Repeat, one round at a time:

1. Read `notes.md`, `results.tsv`, and whatever you need in `experiments/`. Every past experiment's
   `train.py` is in `experiments/<name>/train.py`; the baseline is `baseline_train.py`.
2. Design {BATCH} experiments for this round. Each starts from a `train.py` you choose (the baseline or
   any earlier experiment) and changes it. Balance two kinds:
   - **exploit:** build on the strongest results so far (refine one, or combine ideas that worked
     in different experiments);
   - **explore:** a new direction, usually starting from the baseline, clearly different from
     everything tried so far.
   Explore more in early rounds and exploit more as strong results emerge, but always keep at
   least one explore experiment per round.
3. For each, create `cands/<name>/` with its `train.py` and `idea.md`. In `idea.md`, after the
   title, state the script it starts from (`Parent: <experiment name or baseline>`) and whether it
   is exploit or explore.
4. Run them together: `ar-exp <name1>=cands/<name1>/train.py <name2>=cands/<name2>/train.py ...`
   (new descriptive names each round, e.g. `r05a-routing-on-r04b`).
5. Update `notes.md`: the round's results, what you learned, the best results so far (names and
   scores), and the plan for the next round.

Use all {BATCH} slots every round. Crashed and diverged experiments count toward the budget.
