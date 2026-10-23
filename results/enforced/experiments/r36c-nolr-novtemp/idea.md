# Value-normalized model + novelty temperature
Parent: r35b-nolr-vnorm
Type: exploit

Add the novelty term to the query temperature (q *= 1 + tanh(c f_doc + d nov); nov = 1 - cos(x0_t, sum of
earlier x0), prefix sums via lower-triangular matmul). Confirm: < r35b (0.94174).
