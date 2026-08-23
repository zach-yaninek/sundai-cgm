/**
 * Pillar 3 — what the app has actually learned about this person.
 *
 * Only entries with an OBSERVED outcome teach anything. A past prediction is not
 * evidence; storing one here would let the personalisation layer learn from its
 * own output and drift while appearing to improve.
 */
import { useState } from "react";
import type { HistoryEntry, Meal } from "../api/client";
import { addHistory, clearHistory, usableHistory } from "../lib/storage";
import { mgdl } from "../lib/format";

export default function History({
  history, lastMeal, onChange,
}: {
  history: HistoryEntry[];
  lastMeal: Meal | null;
  onChange: (h: HistoryEntry[]) => void;
}) {
  const [peak, setPeak] = useState("");
  const usable = usableHistory().length;

  const log = () => {
    if (!lastMeal || peak === "") return;
    onChange(addHistory(lastMeal, { peak: Number(peak) }));
    setPeak("");
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h2>What you've logged</h2>
          <p className="sub">
            {usable} meal{usable === 1 ? "" : "s"} with a real outcome.
            {usable > 0 && " Each one sharpens your estimate."}
          </p>
        </div>
        {history.length > 0 && (
          <button type="button" className="link" onClick={() => { clearHistory(); onChange([]); }}>
            clear
          </button>
        )}
      </div>

      {lastMeal && (
        <div className="log-row">
          <label htmlFor="peak">
            What did your glucose actually peak at after that meal?
          </label>
          <div className="input-row">
            <input
              id="peak"
              type="number"
              min={40}
              max={400}
              value={peak}
              onChange={(e) => setPeak(e.target.value)}
              placeholder="e.g. 148"
            />
            <span className="unit">mg/dL</span>
            <button type="button" className="secondary" onClick={log} disabled={peak === ""}>
              Log it
            </button>
          </div>
          <p className="hint">
            Record what actually happened, not what was predicted — that is what
            the model learns from.
          </p>
        </div>
      )}

      {history.length === 0 ? (
        <p className="empty">
          Nothing logged yet. Assess a meal, then come back and record what your
          glucose actually did.
        </p>
      ) : (
        <ul className="history">
          {history.slice().reverse().map((h, i) => (
            <li key={i}>
              <span className="h-meal">
                {Math.round(h.meal?.carbs ?? 0)} g carbohydrate · {h.meal?.meal_type}
              </span>
              <span className="h-outcome">
                {h.observed_peak != null ? `peaked ${mgdl(h.observed_peak)}` : "no outcome recorded"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
