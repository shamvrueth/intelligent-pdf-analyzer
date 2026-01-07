import joblib
from pathlib import Path
from preparecsv import PrepareCSV
from decisiontree import load_and_clean, add_derived_features, build_features_and_target

MODEL_PATH = Path("models/rf_heading5.joblib")  
PROB_THRESHOLD = 0.4
OUTPUT_CSV = "predictions.csv"
OUTPUT_JSON = "predictions.json"

payload = joblib.load(MODEL_PATH)
model = payload["model"]
feature_cols = payload.get("feature_cols", None)
saved_threshold = payload.get("threshold", None)

class PredictAnyPDF():
    def __init__(self, pdf_path, out_path, pdf_name):
        self.pdf_path = pdf_path
        self.out_path = out_path
        self.pdf_name = pdf_name

        self.csv_path = Path(out_path) / f"{self.pdf_name}.csv"
        if not self.csv_path.exists():
            PrepareCSV(pdf_path, out_path) # called only if csv doesn't exist
    
    def create_df(self):
        df = load_and_clean(self.csv_path)
        df = add_derived_features(df)
        return df

    def predict(self, df):
        X, y, groups, feature_cols = build_features_and_target(df)
        probs = model.predict_proba(X)[:, 1]
        threshold = saved_threshold if saved_threshold is not None else PROB_THRESHOLD
        preds = (probs >= threshold).astype(int)
        df = df.copy()
        df["pred_prob"] = probs
        df["pred"] = preds
        predicted = df[df["pred"] == 1][["text", "pred_prob"]]
        print(predicted)
        print(f"\nPredicted headings count: {len(predicted)}")

# pdf_path = "./test/files/collection"
# out_path = "./test/outputs"
# object = PredictAnyPDF(pdf_path, out_path, "South of France - Cuisine")
# df = object.create_df()
# object.predict(df)
