# r88c + n-gram repetition depth (one batched sort for n = 1..4)
Parent: r88c-r87b-dist-nometa. Type: exploit (+ refine).
depth_t = longest n <= 4 such that the n-gram ending at t already occurred in the document (rolling 16-bit
hashes, one batched int32 sort over the 4 key sets). A cheap proxy for the suffix-match length used by
infini-gram style models: a long match means the continuation is likely copyable. Confirm: < 0.923063.
