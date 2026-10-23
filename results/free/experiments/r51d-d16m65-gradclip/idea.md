# proxy: global grad-norm clipping at 2x the running average norm

Parent: r45d. Explore (stability): no clipping so far; clip the summed gradient's global norm to 2x its EMA (after step 50) so loss spikes do not pollute Adam/Muon momenta. All on GPU, no host sync.
