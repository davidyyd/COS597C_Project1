# r32a + input bigram 196613 rows + output trigram table (memory-neutral trade)

Parent: r32a-d21-nosmear-ar2 (= r30a no smear, lean all_reduce). Output trigram table gave -0.0007 at d16 but OOMs at d21 (r29e); shrink the input bigram 262139 -> 196613 rows (-117M params) to pay for the +117M-param trigram_out table.
