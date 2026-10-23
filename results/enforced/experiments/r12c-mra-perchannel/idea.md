# MLP reads the fresh attention output with per-channel weights
Parent: r11a-ms-mra
Type: exploit

The MLP's undiluted read of the attention output uses one scalar beta per layer. With a per-channel vector
beta_l in R^512 (init 0) the MLP can amplify only the channels of the fresh retrieval that carry new
information. Confirm: < r11a (0.94787) by > 0.0005.
