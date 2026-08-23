"""rung3_clip.py — CLIP image embedding -> glucose response.

Does the photograph carry signal the macros miss? This is the offline backend:
no API, no credentials, so it is the first path that can actually ship.

    python embed.py && python rung3_clip.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA

import embed
import evaluate
import rung1_macros
import targets

ARTIFACTS = Path(__file__).parent / "artifacts"
TARGET = "iauc"
N_COMPONENTS = 64

PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.5,
    reg_lambda=2.0, min_child_weight=5,
    objective="reg:squarederror", random_state=0, n_jobs=4,
)


def _fit(X_tr, y_tr, X_te, params=None):
    model = xgb.XGBRegressor(**(params or PARAMS))
    model.fit(X_tr, y_tr, verbose=False)
    return model.predict(X_te)


def clip_raw(X_tr, y_tr, X_te):
    """All 512 dimensions straight into the tree ensemble."""
    cols = [c for c in X_tr.columns if c.startswith("clip_")]
    return _fit(X_tr[cols], y_tr, X_te[cols])


def clip_pca(X_tr, y_tr, X_te):
    """PCA to 64 dimensions, **fitted on the training fold only**.

    Fitting the projection on all rows before splitting would leak test-fold
    structure into the basis. It is a quiet leak — scores improve and nothing
    errors — so the fit lives inside the fold.
    """
    cols = [c for c in X_tr.columns if c.startswith("clip_")]
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(X_tr[cols])
    return _fit(pca.transform(X_tr[cols]), y_tr, pca.transform(X_te[cols]))


def clip_pca_macros(X_tr, y_tr, X_te):
    """PCA-reduced image features concatenated with the logged macros."""
    clip_cols = [c for c in X_tr.columns if c.startswith("clip_")]
    macro_cols = [c for c in X_tr.columns if not c.startswith("clip_")]
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(X_tr[clip_cols])
    tr = np.hstack([pca.transform(X_tr[clip_cols]), X_tr[macro_cols].to_numpy()])
    te = np.hstack([pca.transform(X_te[clip_cols]), X_te[macro_cols].to_numpy()])
    return _fit(tr, y_tr, te)


def build_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Join CLIP vectors onto the modelling set. Returns (X, y, groups, df)."""
    df = targets.modelling_set()
    vectors, keys = embed.load_embeddings()
    lookup = {k: i for i, k in enumerate(keys)}

    have = df["zip_member"].map(lambda m: m in lookup)
    dropped = int((~have).sum())
    if dropped:
        print(f"  {dropped} rows have no cached embedding — run embed.py; skipping them")
    df = df[have].reset_index(drop=True)

    idx = df["zip_member"].map(lookup).to_numpy()
    clip = pd.DataFrame(
        vectors[idx],
        columns=[f"clip_{i:03d}" for i in range(vectors.shape[1])],
        index=df.index,
    )
    macros, _ = rung1_macros.build_features(df)
    X = pd.concat([clip, macros], axis=1)
    return X, df[TARGET].to_numpy(), df["subject"].to_numpy(), df


def main() -> None:
    X, y, groups, df = build_matrix()
    print(f"\nfeature matrix: {X.shape[0]:,} rows x {X.shape[1]} cols "
          f"({sum(c.startswith('clip_') for c in X.columns)} CLIP + "
          f"{sum(not c.startswith('clip_') for c in X.columns)} macro)")

    macro_cols = [c for c in X.columns if not c.startswith("clip_")]
    results, oof_store = [], {}
    for regime in ("cold", "known"):
        contenders = [
            ("global mean (floor)", evaluate.global_mean),
            ("rung 1: macros only", lambda a, b, c: _fit(a[macro_cols], b, c[macro_cols],
                                                         rung1_macros.PARAMS)),
            ("rung 3: CLIP raw 512", clip_raw),
            (f"rung 3: CLIP PCA-{N_COMPONENTS}", clip_pca),
            ("rung 3: CLIP PCA + macros", clip_pca_macros),
        ]
        if regime == "known":
            contenders.insert(1, ("subject mean (floor)",
                                  evaluate.subject_mean_factory(groups)))
        for name, fn in contenders:
            m, oof = evaluate.run(fn, X, y, groups, regime=regime, name=name)
            results.append(m)
            oof_store[(regime, name)] = oof

    evaluate.report(results, target=TARGET)

    # Persist the best CLIP variant, refit on everything.
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "models").mkdir(exist_ok=True)
    clip_cols = [c for c in X.columns if c.startswith("clip_")]
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(X[clip_cols])
    final = xgb.XGBRegressor(**PARAMS)
    final.fit(pca.transform(X[clip_cols]), y, verbose=False)
    final.get_booster().save_model(str(ARTIFACTS / "models" / "rung3_clip.json"))
    np.savez_compressed(
        ARTIFACTS / "models" / "rung3_pca.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
    )

    preds_path = ARTIFACTS / "predictions.parquet"
    add = df[["subject", "timestamp"]].copy()
    for regime in ("cold", "known"):
        add[f"pred_rung3_{regime}"] = oof_store[(regime, f"rung 3: CLIP PCA-{N_COMPONENTS}")]
    evaluate.upsert_predictions(preds_path, add)

    results_path = ARTIFACTS / "results.json"
    existing = json.loads(results_path.read_text()) if results_path.exists() else {}
    existing["rung3"] = {
        "target": TARGET,
        "n": int(len(df)),
        "subjects": int(df.subject.nunique()),
        "embedding": f"{embed.MODEL_NAME}/{embed.PRETRAINED}",
        "n_components": N_COMPONENTS,
        "results": [
            {k: v for k, v in r.items() if k != "mae_ci"} | {"mae_ci": list(r["mae_ci"])}
            for r in results
        ],
    }
    results_path.write_text(json.dumps(existing, indent=2))

    print(f"\nwrote {ARTIFACTS / 'models' / 'rung3_clip.json'}")
    print(f"wrote {ARTIFACTS / 'models' / 'rung3_pca.npz'}")
    print(f"updated {preds_path} and {results_path}")


if __name__ == "__main__":
    main()
