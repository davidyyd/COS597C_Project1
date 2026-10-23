This run uses **linear search**: you improve one model, one experiment at a time. `model.py` is
always the current best model; it starts as the baseline.

Repeat:

1. Read `notes.md`, the end of `results.tsv`, and whatever you need in `experiments/`.
2. Form one research hypothesis. Implement it in `model.py`; write `idea.md` next to it.
3. Check it: `ar-check model.py` (required: `ar-exp` trains only models that passed; fix and
   check again until it passes, or drop the idea).
4. Run it: `ar-exp <name>=model.py` (a new descriptive name each time, e.g. `e012-gated-attn`).
5. Read the result.
   - **Keep** it if its val_loss is lower than the current best: `model.py` stays as it is and this
     experiment is the new best.
   - **Otherwise discard** it: restore the current best with
     `cp experiments/<best name>/model.py model.py` (or `cp baseline_model.py model.py` if nothing
     has beaten the baseline yet). The experiment itself stays in the history.
6. Update `notes.md`: the result, what you learned, the current best (name and score), and what to
   try next.

One experiment per `ar-exp` call. Crashed and diverged experiments count toward the budget; fix a
trivial bug and rerun under a new name only if the idea is still worth testing.
