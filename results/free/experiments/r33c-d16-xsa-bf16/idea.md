# d16 (r30d): learnable XSA, bf16 formulation

Parent: r30d-d16-nosmear (0.78214). Retry of r32d (OOM from fp32 intermediates): y <- y - tanh(alpha_h) * (<y,v>/<v,v>) v computed in bf16 (dot products only are [B,H,T,1]).
