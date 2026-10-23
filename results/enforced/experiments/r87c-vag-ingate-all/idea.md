# MLP input gates in every layer (early layers now document-isolated)
Parent: r83a-vag-dm-ingate
Type: explore
Early MLP input gates hurt badly before (r71a, +0.004). With early layers isolated to the document, the early
residual is cleaner; re-test gates in all layers. Confirm: below r83a (0.94006).
