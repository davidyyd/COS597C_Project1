# d21 no smear + lean all_reduce + empty_cache before first all_reduce + bigram sign trick

Parent: r30a-d21-r28b-seed42 (best, 0.77256) via r32a (crashed: NCCL could not allocate for the first big all_reduce because PyTorch's cache held ~all memory -> torch.cuda.empty_cache() at step 0). Plus bigram sign trick (r32c: -0.0006 at d16).
