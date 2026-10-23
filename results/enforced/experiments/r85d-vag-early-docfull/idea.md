# Early layers: whole current document visible (document mask, no sliding window)
Parent: r83a-vag-dm-ingate
Type: explore
Question: is the early-layer gain from isolation (only the current document) or from locality? Remove the 1024-token
window in the document-masked layers 0-2, so they see the whole current document. Better than r83a (0.94006): early
layers want the document, not locality; worse: locality also matters.
