"""Cross-validate the MLP classifier on PST embeddings, with folds split by CATH superfamily.

Domains of one superfamily are usually homologous, so a superfamily never spans two folds: the
score measures how well the model recognises an architecture in superfamilies it has not seen.
Inside each training fold, 10% of the superfamilies are held out to pick the checkpoint with the
lowest validation loss; the test fold is used only for the reported metrics.

    python -m scripts.train_mlp data/embeddings/pst_t33_so.pt --folds 5 --epochs 100

Restored from the first version of this repository (``mlp_train_eval.py``). Writes
``metrics.json`` (per-fold and mean accuracy and macro F1, out-of-fold confusion matrix) and
``predictions.csv`` to ``--out``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_model import MLPClassifier
from scripts.cath_labels import ARCHITECTURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("embeddings", help="file written by scripts.extract_embeddings")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="logs/mlp_cv")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.embeddings, weights_only=False)
    x = torch.as_tensor(data["embeddings"], dtype=torch.float32)
    y = torch.as_tensor(data["labels"], dtype=torch.long)
    groups = np.asarray(data["superfamily"])

    def loader(idx: np.ndarray, shuffle: bool = False) -> DataLoader:
        return DataLoader(TensorDataset(x[idx], y[idx]), batch_size=args.batch_size, shuffle=shuffle)

    folds = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(folds.split(x, y, groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
        fit_pos, val_pos = next(inner.split(train_idx, groups=groups[train_idx]))
        fit_idx, val_idx = train_idx[fit_pos], train_idx[val_pos]

        model = MLPClassifier(
            input_dim=x.size(1),
            hidden_dim=args.hidden_dim,
            num_classes=len(ARCHITECTURES),
            dropout=args.dropout,
            lr=args.lr,
        )
        checkpoint = ModelCheckpoint(dirpath=out / f"fold_{fold}", monitor="val_loss", mode="min")
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
            logger=CSVLogger(out, name=f"fold_{fold}"),
            deterministic=True,
            enable_model_summary=False,
        )
        trainer.fit(model, loader(fit_idx, shuffle=True), loader(val_idx))

        best = MLPClassifier.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu").eval()
        with torch.no_grad():
            pred = best(x[test_idx]).argmax(dim=1).numpy()
        true = y[test_idx].numpy()
        fold_metrics.append(
            {
                "fold": fold,
                "train": len(fit_idx),
                "val": len(val_idx),
                "test": len(test_idx),
                "accuracy": accuracy_score(true, pred),
                "macro_f1": f1_score(true, pred, average="macro"),
            }
        )
        print(f"fold {fold}: accuracy {fold_metrics[-1]['accuracy']:.4f}, macro F1 {fold_metrics[-1]['macro_f1']:.4f}")
        predictions.append(
            pd.DataFrame(
                {
                    "cath_id": [data["cath_id"][i] for i in test_idx],
                    "superfamily": groups[test_idx],
                    "label": true,
                    "prediction": pred,
                    "fold": fold,
                }
            )
        )

    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    summary = {
        name: {
            "mean": float(np.mean([m[name] for m in fold_metrics])),
            "std": float(np.std([m[name] for m in fold_metrics])),
        }
        for name in ("accuracy", "macro_f1")
    }
    report = {
        "embeddings": str(args.embeddings),
        "args": vars(args),
        "classes": list(ARCHITECTURES.values()),
        "folds": fold_metrics,
        "summary": summary,
        "confusion_matrix": confusion_matrix(
            predictions["label"], predictions["prediction"], labels=range(len(ARCHITECTURES))
        ).tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(
        f"accuracy {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}, "
        f"macro F1 {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
