# DIAGNOSTIC: r09c with a torch.profiler table (steps 100-102) and phase timings (steps 200-299)
Parent: r09c-conv0-all. Type: exploit (diagnostic).

Purpose: find which kernels make the gate/smear additions cost 2-4 ms/step non-additively. Model identical
to r09c; the profiler inflates training_seconds, so the score of this run is not a candidate result.
