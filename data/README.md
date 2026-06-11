# data/

Put Market1501 here (git-ignored):

```
data/market1501/Market-1501-v15.09.15/
├── bounding_box_train/
├── bounding_box_test/
└── query/
```

Automated download: `bash scripts/download_datasets.sh`
Verify: `python scripts/check_dataset_structure.py data`

Market-1501 is for research purposes only; follow the dataset's original terms of use.
