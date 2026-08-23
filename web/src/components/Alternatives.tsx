/**
 * Changes that would lower this person's predicted response.
 *
 * Copy rule, and it is the sharpest one in the product: these are predictions,
 * not dietary advice. "Changes this model predicts would lower your response",
 * never "eat this instead".
 *
 * An empty list is a real answer — the meal is already in their lower range —
 * and is shown as one rather than padded with a suggestion.
 */
import type { AlternativesResponse, Meal } from "../api/client";
import { mgdl, pct } from "../lib/format";

export default function Alternatives({
  data, onApply,
}: {
  data: AlternativesResponse | null;
  onApply: (m: Meal) => void;
}) {
  if (!data) return null;
  // The API returns them gentlest-ask-first, which reads as unsorted once the
  // probabilities are on screen (67%, 67%, 73%). Show most effective first.
  const edits = [...(data.edits ?? [])].sort(
    (a, b) => (a.probability ?? 1) - (b.probability ?? 1),
  );
  const past = data.from_your_history ?? [];

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>What would lower it</h3>
          <p className="sub">{data.note}</p>
        </div>
      </div>

      {edits.length === 0 ? (
        <p className="empty">
          No change was needed — this meal is already in the lower range of what
          the model predicts for you.
        </p>
      ) : (
        <ul className="edits">
          {edits.map((e) => (
            <li key={e.description}>
              <div className="edit-main">
                <strong>{e.description}</strong>
                <span className="edit-delta">
                  {pct(e.probability)} <i>({Math.round((e.delta_probability ?? 0) * 100)} pts)</i>
                </span>
              </div>
              <div className="edit-meta">
                <span>peak {mgdl(e.predicted_peak_mgdl)}</span>
                {Object.entries(e.changes ?? {}).map(([k, v]) => (
                  <span key={k} className="edit-change">
                    {k} {v > 0 ? "+" : ""}{Math.round(v)}
                  </span>
                ))}
                {e.resulting_meal && (
                  <button type="button" className="link" onClick={() => onApply(e.resulting_meal as Meal)}>
                    use this
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      {past.length > 0 && (
        <div className="past">
          <h4>Meals of yours that went well</h4>
          <ul>
            {past.map((p, i) => (
              <li key={i}>
                {Math.round(p.meal?.carbs ?? 0)} g carbohydrate
                {" · "}peaked at {mgdl(p.observed_peak)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
