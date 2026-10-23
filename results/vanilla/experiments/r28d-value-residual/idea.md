# Value residual from the first layer (ResFormer) on top of value embeddings
Parent: r25a-bf16-resid. Type: explore (known method).

Hypothesis: value embeddings already give later layers token identity in their values; the first layer's
values add local context (they are computed from the embedding, mixed by layer-0 attention weights' input).
Tests whether ResFormer's value residual still helps with VEs present.
Change: v_l <- v_l + λ_l v_0 (λ init 0, per layer).
Confirm: < 0.95726. Closest work: ResFormer (Zhou et al. 2024).
