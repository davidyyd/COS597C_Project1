# Best + temporal MLP input gate (gate features of the previous position, cheap form)
Parent: r64a-ingate-instead
Type: exploit

Hypothesis: r71d showed that letting the MLP input gate also see the previous position gives -0.0021 on the fast
base. Here a cheap form (no extra 512-wide projection): the rank-32 gate features z = D u are shifted by one
position and the up-map reads both: u <- u * 2σ(U z_t + U' z_{t-1}). Feature selection becomes context-dependent.
To fit the step cap the gated layers are 5-7 (r64a: 4-7); r71a showed the late layers matter most.
Confirm: below 0.93169 by > 0.0005 and under the time cap.
Closest work: token shift (RWKV), smear; here the shift feeds only a feature-selection gate, not the stream.
