"""annotate.py — estimate meal macros from the photograph.

    python annotate.py --provider local --limit 300
    python annotate.py --provider claude            # needs ANTHROPIC_API_KEY

Two interchangeable providers write the *same* parquet, so nothing downstream
learns which model produced a row:

    image_path · carbs_g · fat_g · protein_g · fiber_g · calories_kcal
    confidence · provider · model_id · annotated_at

Re-running with a different ``--provider`` adds rows rather than overwriting, and
``provider``/``model_id`` travel with every row, so a mixed-provenance file stays
interpretable and the two can be compared on the same photos.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cgm

ARTIFACTS = Path(__file__).parent / "artifacts"
OUT = ARTIFACTS / "vision_macros.parquet"

FIELDS = ["carbs_g", "fat_g", "protein_g", "fiber_g", "calories_kcal"]
SCHEMA = {
    "type": "object",
    "properties": {
        **{f: {"type": "number"} for f in FIELDS},
        "confidence": {"type": "number"},
    },
    "required": FIELDS + ["confidence"],
    "additionalProperties": False,
}

PROMPT = (
    "Estimate the nutritional content of the food in this photograph, for the "
    "portion shown.\n\n"
    "Report grams of carbohydrate, fat, protein and fiber, and total calories. "
    "Give your best single estimate for the whole plate as served, not a range "
    "and not per-ingredient. If several items are visible, sum them.\n\n"
    "Set confidence between 0 and 1: how sure you are of the carbohydrate "
    "figure specifically, since that is what drives blood glucose. Lower it when "
    "the portion size is ambiguous, the food is partly hidden, or you cannot "
    "identify a major component."
)

CLAUDE_MODEL = "claude-opus-5"
LOCAL_MODEL = "qwen2.5vl:7b"


# ------------------------------------------------------------------ sampling

def pick_photos(limit: int | None, *, seed: int = 0) -> pd.DataFrame:
    """Choose which meals to annotate, stratified across subjects and meal types.

    A head() sample would be all subject 1 and mostly breakfast. Stratifying
    keeps the per-subject and per-category error estimates meaningful, which is
    the whole reason a few hundred photos is enough.
    """
    import targets

    df = targets.modelling_set()
    if limit is None or limit >= len(df):
        return df

    # Proportional allocation per (subject, meal_type), at least one from each
    # cell so no subject drops out entirely, then trimmed back to `limit`.
    rng = np.random.default_rng(seed)
    picks = []
    for _, cell in df.groupby(["subject", "meal_type"], sort=True):
        n = max(1, round(limit * len(cell) / len(df)))
        picks.append(cell.sample(min(n, len(cell)),
                                 random_state=int(rng.integers(1 << 31))))
    take = pd.concat(picks, ignore_index=True)
    if len(take) > limit:
        take = take.sample(limit, random_state=seed)
    return take.reset_index(drop=True)


# ------------------------------------------------------------------ providers

def annotate_local(rows: pd.DataFrame, *, model: str = LOCAL_MODEL) -> list[dict]:
    """Estimate macros with a local VLM through Ollama. No credentials, no cost."""
    import ollama

    out = []
    started = time.time()
    for i, row in enumerate(rows.itertuples(), start=1):
        blob = cgm.photo_bytes(row.subject, row.image_path)
        if blob is None:
            continue
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": PROMPT, "images": [blob]}],
                format=SCHEMA,
                options={"temperature": 0},
            )
            data = json.loads(resp["message"]["content"])
        except Exception as exc:  # a single bad photo must not kill the run
            print(f"  ! {row.image_path}: {type(exc).__name__}: {exc}")
            continue
        out.append(_record(row, data, "local", model))
        if i % 20 == 0:
            rate = (time.time() - started) / i
            print(f"  {i:>4}/{len(rows)}  {rate:.1f}s/img  "
                  f"eta {rate * (len(rows) - i) / 60:.0f}min")
    return out


def annotate_claude(rows: pd.DataFrame, *, model: str = CLAUDE_MODEL) -> list[dict]:
    """Estimate macros with Claude via the Batches API (50% cost, async)."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    keyed, requests = {}, []
    for n, row in enumerate(rows.itertuples()):
        blob = cgm.photo_bytes(row.subject, row.image_path)
        if blob is None:
            continue
        cid = f"m{n:05d}"
        keyed[cid] = row
        requests.append(Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=1024,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA},
                               "effort": "low"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(blob).decode()}},
                    {"type": "text", "text": PROMPT},
                ]}],
            ),
        ))

    batch = client.messages.batches.create(requests=requests)
    print(f"  batch {batch.id} submitted with {len(requests)} requests")
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  {batch.processing_status}: {batch.request_counts}")
        time.sleep(30)

    out = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        row = keyed[result.custom_id]
        text = next(b.text for b in result.result.message.content if b.type == "text")
        out.append(_record(row, json.loads(text), "claude", model))
    return out


def _record(row, data: dict, provider: str, model_id: str) -> dict:
    return {
        "image_path": row.image_path,
        "subject": row.subject,
        "timestamp": row.timestamp,
        **{f: float(data.get(f, np.nan)) for f in FIELDS},
        "confidence": float(data.get("confidence", np.nan)),
        "provider": provider,
        "model_id": model_id,
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=["local", "claude"], default="local")
    ap.add_argument("--limit", type=int, default=300,
                    help="photos to annotate; omit for all (slow on local)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    rows = pick_photos(args.limit)
    print(f"\nannotating {len(rows)} photos with provider={args.provider}")
    print(f"  {rows.subject.nunique()} subjects, "
          f"meal types {rows.meal_type.value_counts().to_dict()}\n")

    if args.provider == "local":
        records = annotate_local(rows, model=args.model or LOCAL_MODEL)
    else:
        records = annotate_claude(rows, model=args.model or CLAUDE_MODEL)

    if not records:
        print("no annotations produced")
        return

    fresh = pd.DataFrame(records)
    ARTIFACTS.mkdir(exist_ok=True)
    if OUT.exists():
        prior = pd.read_parquet(OUT)
        # Idempotent per (image, provider): a re-run replaces its own rows and
        # leaves the other provider's alone, so the two stay comparable.
        keep = ~(prior["image_path"].isin(fresh["image_path"])
                 & prior["provider"].isin(fresh["provider"]))
        fresh = pd.concat([prior[keep], fresh], ignore_index=True)
    fresh.to_parquet(OUT, index=False)

    print(f"\nwrote {OUT}  ({len(fresh):,} rows total)")
    print(fresh.groupby("provider").size().to_string())


if __name__ == "__main__":
    main()
