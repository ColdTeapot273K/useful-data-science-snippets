# INFO: Example for dataset versioning snippet; means "2021/01/29" in YYYY/MM/DD
DATASET_BUILD_VERSION = 20210129

# INFO: Example default LightGBM config, for use in canary tests - runs for <30 sec
DEFAULT_MODEL_CONFIG = {
    "boosting_type": "gbdt",
    "class_weight": None,
    "colsample_bytree": 1.0,
    "importance_type": "split",
    "learning_rate": 0.1,
    "max_depth": -1,
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 100,
    "n_jobs": -1,
    "num_leaves": 31,
    "objective": None,
    "random_state": None,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    # "silent": True,
    "silent": False,
    "subsample": 1.0,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
}
