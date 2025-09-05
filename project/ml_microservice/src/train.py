from __future__ import annotations
import argparse, os, json
import joblib, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .pipeline import build_pipeline
from .utils import (
    load_config, load_dataset, make_split, cv_strategy_from_config,
    ensure_dir, save_split_indices, compute_metrics_binary,
    plot_and_save_roc, plot_and_save_pr, plot_and_save_confusion
)

# --- Adaptateur: texte -> DataFrame -> pipeline tabulaire ---
class TextToTabularAdapter:
    """
    Attend en entrée une liste de chaînes.
    Chaque chaîne doit être soit:
      - un JSON object (ex: '{"tenure":5,"gender":"Female",...}')
      - ou du 'k=v' séparé par des virgules (ex: 'tenure=5, gender=Female, ...')
    """
    def __init__(self, inner_pipeline):
        self.inner = inner_pipeline

    @staticmethod
    def _parse_text(s: str) -> dict:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # fallback simple k=v,k=v
        items = [p.strip() for p in s.split(",") if p.strip()]
        out = {}
        for it in items:
            if "=" in it:
                k, v = it.split("=", 1)
                out[k.strip()] = v.strip()
        if not out:
            raise ValueError("Input text must be JSON object or 'k=v' comma-separated string")
        return out

    def _to_df(self, texts):
        rows = [self._parse_text(t) for t in texts]
        return pd.DataFrame(rows)

    def predict(self, texts):
        X = self._to_df(texts)
        return self.inner.predict(X)

    def predict_proba(self, texts):
        if not hasattr(self.inner, "predict_proba"):
            raise AttributeError("inner model has no predict_proba")
        X = self._to_df(texts)
        return self.inner.predict_proba(X)

    @property
    def classes_(self):
        mdl = getattr(self.inner, "named_steps", {}).get("model") if hasattr(self.inner, "named_steps") else None
        return getattr(mdl, "classes_", None)

def _prefix_params(params): return {f"model__{k}": v for k, v in params.items()}

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    artifacts_dir = cfg.outputs.get("artifacts_dir", "artifacts")
    ensure_dir(artifacts_dir)

    df = load_dataset(cfg.data["csv_path"])
    target = cfg.data["target"]

    # Harmonisation binaire Yes/No -> 1/0
    if df[target].dtype == object:
        s = df[target].astype(str).str.strip()
        if set(s.unique()) <= {"Yes", "No"}: df[target] = (s == "Yes").astype(int)
        elif set(s.unique()) <= {"True", "False"}: df[target] = (s == "True").astype(int)

    X_train, X_test, y_train, y_test, train_idx, test_idx = make_split(
        df, target, cfg.data["test_size"], cfg.data["random_state"]
    )

    pipe = build_pipeline(cfg.features["numeric"], cfg.features["categorical"], cfg.model["type"])

    cv = cv_strategy_from_config(cfg.cv)
    scoring = cfg.cv.get("scoring", "roc_auc")
    n_jobs = int(cfg.cv.get("n_jobs", -1))
    param_grid = _prefix_params(cfg.model.get("params", {}))

    mlflow.sklearn.autolog(log_models=False)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if exp_name: mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tags({"model_type": cfg.model["type"], "dataset_path": cfg.data["csv_path"]})
        split_path = save_split_indices(artifacts_dir, train_idx, test_idx); mlflow.log_artifact(split_path)

        gs = GridSearchCV(pipe, param_grid if param_grid else {}, scoring=scoring, cv=cv, n_jobs=n_jobs, refit=True)
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        mlflow.log_params({"best_params": gs.best_params_})
        mlflow.log_metric("cv_best_score", float(gs.best_score_))

        # Test set
        y_proba = None
        if hasattr(best, "predict_proba"):
            p = best.predict_proba(X_test)
            if p.shape[1] == 2: y_proba = p[:, 1]
        y_pred = best.predict(X_test)

        metrics = compute_metrics_binary(y_test, y_proba, y_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # Artefacts
        roc = os.path.join(artifacts_dir, "roc_curve.png")
        pr = os.path.join(artifacts_dir, "pr_curve.png")
        cm = os.path.join(artifacts_dir, "confusion_matrix.png")
        plot_and_save_roc(y_test, y_proba, roc); mlflow.log_artifact(roc)
        plot_and_save_pr(y_test, y_proba, pr); mlflow.log_artifact(pr)
        plot_and_save_confusion(y_test, y_pred, cm); mlflow.log_artifact(cm)

        preds = os.path.join(artifacts_dir, "predictions.csv")
        pd.DataFrame({"index": X_test.index, "y_true": y_test.values, "y_pred": y_pred, "y_proba": y_proba}).to_csv(preds, index=False)
        mlflow.log_artifact(preds)

        # Log du pipeline "pur" dans MLflow (pour registry/éval)
        registered_name = cfg.model.get("register_name")
        mlflow.sklearn.log_model(best, artifact_path="model",
                                 registered_model_name=registered_name if registered_name else None)

        # ⚠️ Sauvegarde du **wrapper** pour l'API immuable app.py
        api_model_path = os.path.join(artifacts_dir, "model.joblib")   # <- app.py attend ce chemin
        joblib.dump(TextToTabularAdapter(best), api_model_path)
        mlflow.log_artifact(api_model_path)

        print(f"[MLflow] run_id={run.info.run_id} test_roc_auc={metrics.get('roc_auc')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
