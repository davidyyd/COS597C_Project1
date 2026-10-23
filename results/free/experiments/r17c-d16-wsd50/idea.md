# d16 (LR 1e-2, warmup 2%): WSD schedule, constant LR to 50% then linear decay to 0

Parent: r16c-d16-warmup2. Exploit (schedule): shorter warmup helped (more time at high LR); try warmup-stable-decay with the decay over the last 50% of steps.
