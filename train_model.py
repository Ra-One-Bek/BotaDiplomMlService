import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATASET_PATH = os.path.join("data", "training_dataset.csv")
MODEL_PATH = os.path.join("models", "career_direction_model.pkl")

FEATURE_COLUMNS = [
    "direction.it",
    "direction.medicine",
    "direction.education",
    "direction.business",
    "direction.creative",
    "direction.engineering",
    "thinkingStyle.analytic",
    "thinkingStyle.creative",
    "thinkingStyle.practical",
    "temperament.introvert",
    "temperament.extrovert",
    "studyProfile.stem",
    "studyProfile.humanities",
    "values.stability",
    "values.income",
    "values.helping",
    "values.freedom",
    "anti.it",
    "anti.medicine",
    "anti.education",
    "anti.business",
    "anti.creative",
    "anti.engineering",
]

TARGET_COLUMN = "target_direction"


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("clf", LogisticRegression(max_iter=2000))
        ]
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": sorted(y.unique().tolist())
        },
        MODEL_PATH
    )

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()