# proxy backin5-mix without the final backout subtraction (ablation)

Parent: r67a-proxy-backin5-mix (0.76388). Now that the last attention layers read x_backout, check whether the final x - lambda*x_backout subtraction still helps (lambda forced to 0).
