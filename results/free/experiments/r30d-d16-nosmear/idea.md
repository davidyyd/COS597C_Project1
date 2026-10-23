# d16 (r28d): remove smear (redundant with the bigram embedding?)

Parent: r28d-d16-trigramout. Explore/simplify: smear mixed the previous token's embedding in; the hashed bigram embedding now carries (t-1, t) information directly. If neutral, removing it simplifies; if positive, it was interfering.
