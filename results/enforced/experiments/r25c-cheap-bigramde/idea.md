# Cheap lineage with the non-linear bigram embedding instead of the unigram deep embedding
Parent: r24b-cheap-deepemb
Type: exploit

On baseline the bigram version beat the unigram one by 0.0006 (r24d vs r23d). Swap it in.
Confirm: < r24b by > 0.0005.
