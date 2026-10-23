# r18b + document masking with per-batch block masks built in-graph
Parent: r18b-r17a-bf16. Type: exploit.

Rows pack several documents; r05b showed hard document masking helps (-0.0022 on the baseline) but its
eager create_block_mask cost 12 ms/step. Here the block masks are built from block-level document ranges
with a few tensor ops inside torch.compile (BlockMask.from_kv_blocks, full blocks marked), which should
cost little and make attention sparser (faster) in the long layers. All attention layers use flex.
Verified on CPU against a dense reference (max err 0 for windows None/128/512).
Confirm: < 0.9369 at ≤ 261 s.
Closest work: intra-document masking (Llama 3, modded-nanogpt). Known; used here as an efficient base.
