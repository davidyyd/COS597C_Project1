# d18w1792 attnscale, weight decay 0.05 (+ Muon timing breakdown)

Parent: r07a-d18w1792-attnscale. Exploit: WD 0.2 hurt at small scale; at 5 tokens/param the model is undertrained, try weaker WD 0.05. Also times Muon's per-shape compute and gather wait (Muon = 195 ms of 868 ms/step in r07a, far above its FLOPs).
