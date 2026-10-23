# Complexity review of a language-model architecture

You review one candidate architecture produced by an automated research agent. The agent starts from
a fixed baseline model and may change only the model file. Your only job: judge whether the
candidate's **complete change relative to the baseline** is a simple, natural method (the kind a
strong paper at a top venue would present as its contribution) or an accumulation of tweaks.

You do not see any results, and you must not guess them: a change can be effective and still too
complex. Judge the method, not its likely score.

## Inputs

### The baseline model (comments removed)

```python
{BASELINE_CODE}
```

### The candidate model (comments removed)

```python
{CANDIDATE_CODE}
```

### The diff, baseline -> candidate

```diff
{DIFF}
```

### Mechanical statistics of the diff

{STATS}

### The author's description (`idea.md`)

This is the author's claim, written to persuade. Use it only to understand intent; judge what the
code actually does.

```markdown
{IDEA}
```

## How to review

1. **List the mechanisms.** A mechanism is one change to what the model computes, with one purpose,
   that can be stated as one formula or one sentence. Count as part of a mechanism: its parameters,
   their initialisation, and the plumbing it needs. Count as one: the same rule applied uniformly in
   several places (every layer, every head). Count separately: changes that serve different
   purposes, even if introduced together; and any special case attached to a mechanism (a different
   rule for some layers, heads or positions). Ignore changes that do not change the function
   (refactoring, caching, renaming, reshaping without effect).
2. **Look for red flags**, each named with the line(s) it is on:
   - hand-set constants without a principled origin (e.g. magic multipliers, hash primes used as
     model components, thresholds), other than standard ones (1/sqrt(d), RMSNorm eps, ...);
   - per-layer, per-head or per-position special cases chosen by hand (e.g. "layers 0-7 do X,
     layers 8-11 do Y"), or the number of something chosen by hand without a principle (e.g. "taps
     at t-1, t-2, t-3");
   - mechanisms that only make sense as patches to other mechanisms in the candidate;
   - several unrelated mechanisms with no common idea;
   - learned scalars or gates added on top of each other at the same place;
   - anything you could not explain to a reader in a short methods section.
3. **Write the method as a paper would.** In at most 3 sentences, state the candidate's method as its
   contribution. If you cannot do that without a list of unrelated parts, it is not one method.
4. **Score `complexity` from 1 to 10** for the whole change relative to the baseline:
   - 1-2: one formula-level change at one place, with at most a few new learned scalars or vectors
     and no new conditioning or plumbing;
   - 3-4: one mechanism with its supporting parts: the formula plus what it needs to work (its own
     learned parameters, what they are conditioned on, bounds and initialisation, a tensor threaded
     to where it is used), still stated in one sentence;
   - 5-6: two or three mechanisms that serve one idea, or one mechanism with arbitrary choices
     (hand-set constants, hand-picked layers, a hand-picked number of parts);
   - 7-8: several mechanisms with separate purposes, or one mechanism with several special cases;
   - 9-10: a pile of changes with no common idea.
   The kind of change does not matter, only its structure. Add to the score for red flags that are
   central to the method; do not subtract for good results (you do not see any).

## Output

Only the JSON object (the schema is enforced):

- `mechanisms`: every mechanism, each as {`name`, `formula`, `purpose`, `where` (lines)}
- `n_mechanisms`: the number of mechanisms listed
- `red_flags`: list of {`flag`, `where` (lines)}
- `method_summary`: at most 3 sentences, as a paper would state the method
- `complexity`: 1-10
- `reasons`: 2-4 sentences shown to the author: what makes the change as complex as it is, and what
  could be removed or unified to make it simpler
