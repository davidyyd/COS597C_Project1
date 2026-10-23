# Only normalize the value embedding (ablation)
Parent: baseline
Type: explore (ablation)

Hypothesis behind the value-path gain: baseline VE tables start at ~1/sqrt(512) of the projected values' scale
and must grow through Adam before they matter. Test the minimal fix: v = W_v u + gate * norm(VE) (gate as in the
baseline). If most of r43a's gain comes from here, the mechanism is the VE scale mismatch.
