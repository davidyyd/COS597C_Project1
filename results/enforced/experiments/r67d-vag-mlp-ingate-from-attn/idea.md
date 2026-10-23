# MLP input gate computed from the attention output
Parent: r58b-va-gates-adam (0.94618)
Type: explore

The MLP input gate from the input itself helped (r63c, -0.0018); a write gate from the attention output hurt
(r61d). Test feature selection driven by what attention just retrieved: u <- u * 2 sigmoid(U V norm(a)). Compare with r63c.
