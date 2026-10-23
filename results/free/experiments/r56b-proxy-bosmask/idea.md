# proxy: BOS targets excluded from the training loss

Parent: r45d-d16m65-canon2-stash (0.76588). The eval ignores zero-byte targets (BOS = document boundaries, unpredictable); stop spending training signal on predicting them (ignore_index=BOS in the summed training CE).
