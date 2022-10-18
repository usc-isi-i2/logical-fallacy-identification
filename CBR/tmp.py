import joblib


# for split in ["train", "dev", "test"]:
for split in ["train"]:
    a = joblib.load(f"gcn_similarities_masked_articles_{split}.joblib")
    a = {
        key.strip(): {
            nested_key.strip(): nested_value
            for nested_key, nested_value in value.items()
        }
        for key, value in a.items()
    }
