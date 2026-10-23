# r29a + lag-2 key smear
Parent: r29a-r27b-headchan-gain. Type: exploit.

Removing the key smear costs 0.003 (r30b): keys that know their predecessor are central (induction).
A second lag lets a key encode its 2-token prefix, so a head can match longer repeated prefixes.
Change: k_t <- k_t + a1 ⊙ k_{t-1} + a2 ⊙ k_{t-2} (a2 per channel, init 0, packed).
Confirm: < 0.935361 within the cap.
