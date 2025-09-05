from __future__ import annotations
import argparse, os, tempfile
import mlflow, mlflow.sklearn
import pandas as pd
from .utils import (
    load_config, load_dataset, load_split_indices, ensure_dir,
    compute_metrics_binary, plot_and_save_roc, plot_and_save_pr, plot_and_save_confusion
)

def _latest_finished_run_id(experiment_name: str | None) -> str | None:
    if experiment_name:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp: return None
        exp_ids = [exp.experiment_id]
    else:
        exp_ids = None
    df = mlflow.search_runs(
        experiment_ids=exp_ids,
        filter_string="attributes.status = 'FINISHED'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if df.empty: return None
    return df.iloc[0]["run_id"]

def main(cfg_path: str, model_uri: str | None):
    cfg = load_config(cfg_path)
    artifacts_dir = cfg.outputs.get("artifacts_dir", "artifacts")
    ensure_dir(artifacts_dir)

    if not model_uri:
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
        run_id = _latest_finished_run_id(exp_name)
        if not run_id: raise RuntimeError("Aucun run MLflow trouvé. Lance d'abord l'entraînement.")
        model_uri = f"runs:/{run_id}/model"
        client = mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as td:
            split_local = client.download_artifacts(run_id, "split.json", td)
            train_idx, test_idx = load_split_indices(split_local)
    else:
        split_local = os.path.join(artifacts_dir, "split.json")
        if not os.path.exists(split_local):
            raise RuntimeError("split.json introuvable. Relance train.py ou fournis --model-uri d'un run.")
        train_idx, test_idx = load_split_indices(split_local)

    model = mlflow.sklearn.load_model(model_uri)

    df = load_dataset(cfg.data["csv_path"])
    target = cfg.data["target"]
    if df[target].dtype == object:
        s = df[target].astype(str).str.strip()
        if set(s.unique()) <= {"Yes", "No"}: df[target] = (s == "Yes").astype(int)
        elif set(s.unique()) <= {"True", "False"}: df[target] = (s == "True").astype(int)

    X = df.drop(columns=[target]); y = df[target]
    X_test = X.loc[test_idx]; y_test = y.loc[test_idx]

    y_proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_test)
        if p.shape[1] == 2: y_proba = p[:, 1]
    y_pred = model.predict(X_test)

    metrics = compute_metrics_binary(y_test, y_proba, y_pred)

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics({f"eval_{k}": float(v) for k, v in metrics.items()})
        roc = os.path.join(artifacts_dir, "eval_roc_curve.png")
        pr = os.path.join(artifacts_dir, "eval_pr_curve.png")
        cm = os.path.join(artifacts_dir, "eval_confusion_matrix.png")
        plot_and_save_roc(y_test, y_proba, roc); mlflow.log_artifact(roc)
        plot_and_save_pr(y_test, y_proba, pr); mlflow.log_artifact(pr)
        plot_and_save_confusion(y_test, y_pred, cm); mlflow.log_artifact(cm)
        preds = os.path.join(artifacts_dir, "eval_predictions.csv")
        pd.DataFrame({"index": X_test.index, "y_true": y_test.values, "y_pred": y_pred, "y_proba": y_proba}).to_csv(preds, index=False)
        mlflow.log_artifact(preds)
        print(f"[Eval] {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-uri", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.model_uri)
