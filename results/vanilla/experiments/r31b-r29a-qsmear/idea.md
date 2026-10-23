# r29a + query smear (bigram queries)
Parent: r29a-r27b-headchan-gain. Type: exploit.

With smeared keys a key at s+1 encodes (token s, token s+1). A query that also encodes its predecessor
(q_t + b ⊙ q_{t-1}) can match on the pair (token t-1, token t) against keys encoding (token s-1, token s):
2-gram-conditioned copying in one head.
Change: q_t <- q_t + b ⊙ q_{t-1} (b per head-channel, init 0, packed). Confirm: < 0.935361 within the cap.
