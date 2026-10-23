# Document-start vocabulary prior
Parent: r83a-vag-dm-ingate
Type: explore
Hypothesis: the first tokens of a document follow a distinct distribution (titles, headers, capitalised words); the
model only knows it is at a document start through attention. A learned vocabulary bias weighted by (1 - f_doc)
(f_doc = log(1+t-doc_start)/log T) gives this prior directly at the output.
Change: logits += (1 - f_doc) b, b in R^vocab (head component), init 0.
Confirm: below r83a by > 0.0005.
