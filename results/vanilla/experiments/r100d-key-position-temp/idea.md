# Key-position attention temperature (control)
Parent: r91d-rowlen-temperature (on r25a-bf16-resid). Type: explore.
Adds k_s *= 1 + b_h log(1+s)/log(T) to the query-position temperature. With QK-norm, a key's norm sets how
sharply it can be selected; scaling by log key position gives each head a learned, content-scaled preference
for early vs late keys (unlike ALiBi's additive distance bias, which was very negative here).
Confirm: < 0.954847.
