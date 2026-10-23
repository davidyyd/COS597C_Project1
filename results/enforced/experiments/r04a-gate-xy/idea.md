# Head gate from retrieved value and query-side input
Parent: r03b-qtanh-prevtok
Type: exploit

The self-gate (gate = 2 sigmoid(a_h . y_h)) decides from what the head retrieved. A query-side term lets it
also decide from what the token is looking for: gate = 2 sigmoid(a_h . y_h + w_h . x). Tests whether the
x-based gated attention (Qiu 2025) is complementary to the self-gate. Confirm: < r03b (0.95282) by > 0.001.
