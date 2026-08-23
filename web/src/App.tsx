/**
 * Vertical slice, not a finished app.
 *
 * It fetches /api/meta, runs one real assessment against the backend and renders
 * the result — enough to prove the contract works end to end before any UI work
 * starts. Replace the body with the real screens; keep the meta fetch and the
 * consent gate.
 */
import { useEffect, useState } from "react";
import { ApiError, assess, getMeta, describeRisk } from "./api/client";
import type { AssessResponse, Meta } from "./api/client";
import { loadLabs } from "./lib/storage";

const DEMO_MEAL = {
  carbs: 66,
  protein: 20,
  fat: 18,
  fiber: 4,
  calories: 712,
  meal_type: "dinner" as const,
  amount_consumed: 100,
};

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [result, setResult] = useState<AssessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMeta()
      .then(async (m) => {
        setMeta(m);
        setResult(await assess({ labs: loadLabs(), meal: DEMO_MEAL, history: [] }));
      })
      .catch((e: unknown) =>
        setError(e instanceof ApiError ? e.message : String(e)),
      );
  }, []);

  if (error) {
    return (
      <main className="wrap">
        <h1>Backend unreachable</h1>
        <pre className="error">{error}</pre>
      </main>
    );
  }
  if (!meta || !result) return <main className="wrap">Loading…</main>;

  return (
    <main className="wrap">
      <h1>Meal response</h1>

      <section className="card">
        <p className="risk">{describeRisk(result.exceeds_140!, meta)}</p>
        <dl className="grid">
          <div>
            <dt>Predicted peak</dt>
            <dd>{result.predicted_peak_mgdl} mg/dL</dd>
          </div>
          <div>
            <dt>Predicted iAUC</dt>
            <dd>{result.predicted_iauc} mg/dL·h</dd>
          </div>
          <div>
            <dt>Confidence</dt>
            <dd>{result.confidence?.band}</dd>
          </div>
          <div>
            <dt>Meals logged</dt>
            <dd>{result.personalization?.meals_logged}</dd>
          </div>
        </dl>
      </section>

      <p className="footnote">
        Model fitted to {meta.cohort?.n_subjects} adults in one study
        {" · "}flag AUC {meta.performance?.auc_with_glucose} with a glucose reading,
        {" "}{meta.performance?.auc_without_glucose} without
        {" · "}version {result.model_version}
      </p>
      <p className="footnote">{meta.disclaimer?.text}</p>

      <p className="todo">
        Replace this slice with the real screens — see <code>web/AGENTS.md</code>.
      </p>
    </main>
  );
}
