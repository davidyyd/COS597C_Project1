# Hard document masking, block masks built eagerly (retry of r04b)
Parent: baseline. Type: explore.

Same hypothesis as r04b-docmask (cross-document attention in packed rows is noise). r04b trained but
crashed at evaluation (illegal memory access while autotuning the compiled mask creation under no_grad);
here create_block_mask runs outside torch.compile (torch.compiler.disable).
Closest work: intra-document masking (Llama 3, modded-nanogpt). Not novel by itself; tests a base.
