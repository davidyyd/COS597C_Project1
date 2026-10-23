# d16w1792 + smear + fast Muon all_gather

Parent: r05a-d16w1792-unet-nest (+ smear from r05d-smear). Exploit. Profile of r05b showed Muon's per-chunk list all_gather (NCCL LL broadcast) costs ~100 of 745 ms/step; replaced by flat async all_gather_into_tensor per chunk (waits at the end, overlaps NS compute). Same math. Expect ~10% faster + smear gain.
