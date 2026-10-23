This run uses **batched linear search**: you improve one model in rounds of {BATCH} experiments
trained in parallel. `model.py` is always the current best model; it starts as the baseline.

Repeat, one round at a time:

1. Read `notes.md`, the end of `results.tsv`, and whatever you need in `experiments/`.
2. Design {BATCH} different research hypotheses, each a change to the current best model. For each,
   create `cands/<name>/` with its own `model.py` (a copy of the current best, then changed) and
   `idea.md`. Make the ideas clearly different from each other.
3. Check each: `ar-check cands/<name>/model.py` (required: `ar-exp` trains only models that passed;
   fix and check again, or replace the idea, until all {BATCH} pass).
4. Run them together: `ar-exp <name1>=cands/<name1>/model.py <name2>=cands/<name2>/model.py ...`
   (new descriptive names each round, e.g. `r05a-value-embeddings`).
5. Read the results. If the best experiment of the round has a lower val_loss than the current
   best, **keep** it: `cp experiments/<that name>/model.py model.py`. Otherwise keep the current
   best. Every experiment stays in the history either way.
6. Update `notes.md`: the round's results, what you learned, the current best (name and score),
   and what to try next.

Use all {BATCH} slots every round. Crashed and diverged experiments count toward the budget.
