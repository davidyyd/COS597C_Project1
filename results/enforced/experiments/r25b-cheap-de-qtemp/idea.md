# Cheap lineage with deep embedding + content query temperature
Parent: r24b-cheap-deepemb
Type: exploit

Add the content term to the doc temperature (q *= 1 + tanh(W x + c f_doc)). Confirm: < r24b by > 0.0005.
