# r83a - ctx-x0 conv + document BIGRAM occurrence count (cheap int32 sort)
Parent: r84b-r83a-noctx-mlpgate2 with MLP gates in layers 4-7 (= r83a without ctx-x0). Type: exploit.
On the control the bigram count beat the unigram count (r86c -0.0021 vs r85d -0.0015). In the stack the
unigram count gave -2e-4 but ran 0.45 s over (r86a). Here: bigram key hashed to 16 bits so that the
(doc, key) sort runs on int32 (cheaper). Confirm: < 0.923372 within 261.66 s.
