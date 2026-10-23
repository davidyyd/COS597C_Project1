# MLP reads this block's attention output directly
Parent: baseline
Type: explore

Hypothesis: r09d showed MLPs gain from an undiluted copy of the token identity. Likewise the information
just retrieved by attention is a small part of norm(x + a) once the residual is large. Give the MLP a
normalized direct copy: u = norm(x + a) + beta_l norm(a), beta init 0.
Confirm: < baseline by > 0.001.
Closest work: parallel attention/MLP blocks (GPT-J: MLP reads x only), DenseFormer. Differs: the MLP
reads the fresh attention output re-normalized, in addition to the stream.
