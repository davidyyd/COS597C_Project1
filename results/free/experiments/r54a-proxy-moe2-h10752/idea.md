# proxy with a 2-expert top-1 MoE MLP (expert hidden 10752, capacity 1.1)

Parent: r45d-d16m65-canon2-stash. Explore: MLP width has been the strongest lever but costs time; a top-1 MoE with 2 experts of hidden 10752 (6x) doubles MLP params at ~1.1x the FLOPs of a dense 6x MLP. Static-capacity routing (index_copy/gather + bmm), Switch balance loss 0.01, router in Muon, experts 'muon per slice'. Compare with the dense 6x control r54b.
