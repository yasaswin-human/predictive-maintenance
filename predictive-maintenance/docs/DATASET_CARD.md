# Dataset Card - AI4I Predictive Maintenance

## Source
UCI Machine Learning Repository, dataset 601.
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Size
~10,000 rows. Identifiers, machine type, sensor measurements, binary
failure label, five failure-mode indicators.

## Limitations
- Synthetic / simulated generation (per UCI description).
- Severe class imbalance (~3% positive).
- Failure-mode columns reveal the label - must be excluded from features.

## Leakage rule
Never use `UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`,
`OSF`, `RNF` as model inputs.

## Preprocessing applied
Median impute + standardize numeric features; most-frequent impute +
one-hot encode `Type`. All transformations fit inside the model pipeline
on the training split only.