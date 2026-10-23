# r83a - ctx-x0 conv + document occurrence-count embedding
Parent: r84b-r83a-noctx-mlpgate2 (MLP gates back to layers 4-7, i.e. r83a without the ctx-x0 conv). Type: exploit.
r85d: the exact causal count of earlier occurrences of x_t in the document (bucketed) gave -0.0015 on the
control at ~+1.5 s. Fund it by the ctx-x0 conv (worth 2.7e-4, 1.5 s). Confirm: < 0.923372 within the cap.
