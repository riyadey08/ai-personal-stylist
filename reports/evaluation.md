# Evaluation report

## feature_parity

- `passed`: True
- `features`: 43

## retrieval_group_split

- `n_train`: 4671
- `n_test`: 1149
- `recall@5`: 0.9974
- `mrr@5`: 0.9973
- `ndcg@5`: 0.9954
- `unique_identities@5`: 3.17
- `duplicate_rate@5`: 0.3664
- `self_match_rate`: 0.0
- `note`: Relevance proxy = same body_shape label; label is 99.3% one class in bundled data.

## body_shape

- `n`: 5820
- `accuracy`: 0.3275
- `macro_f1`: 0.1677
- `majority_baseline_accuracy`: 0.993
- `per_class_recall`: {'Inverted Triangle': 0.3284, 'Pear': 1.0, 'Rectangle': 0.0}
- `confusion_matrix`: {'labels': ['Inverted Triangle', 'Pear', 'Rectangle'], 'matrix': [[1898, 1810, 2071], [0, 8, 0], [0, 33, 0]]}
