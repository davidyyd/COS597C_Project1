# No time-shifted reads in the isolated layers 0-2; MLP input gate also in layer 3
Parent: r84b-noshift012-lr1357
Type: exploit
r84b (drop the key shift read in layers 0-2, low-rank gate back in layer 1) is the best (0.92944). Also drop the MLP's
read of a[t-1] in layers 0-2 and spend the time on an MLP input gate in layer 3.
Confirm: below r84b by > 0.0003.
