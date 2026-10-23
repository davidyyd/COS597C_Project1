# Value embeddings on the last four layers
Parent: r43a-base-valuepath
Type: explore

Moving the (normalized) value embeddings from layers 1,3,5,7 to 0,2,4,6 cost +0.0062 (r47d): token identity in
values seems to matter most late (the last layer). Test placing all four on the last four layers 4,5,6,7.
Confirm: < r43a (0.94690).
