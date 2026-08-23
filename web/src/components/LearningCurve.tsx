/**
 * How much logging actually buys — measured, not promised.
 *
 * These points come from a leave-one-subject-out sweep on held-out people: the
 * population model is trained without a subject, then k of their meals are
 * revealed and it is scored on the rest. Read from `/api/meta` rather than
 * hardcoded, so it stays true when the model is retrained.
 */
import type { Meta } from "../api/client";
import Chart from "./Chart";

export default function LearningCurve({
  meta, mealsLogged,
}: {
  meta: Meta;
  mealsLogged: number;
}) {
  const points = (meta.performance?.learning_curve ?? [])
    .map((p) => ({ x: p.meals_logged ?? 0, y: p.mae ?? 0 }));
  if (points.length < 2) return null;

  const start = points[0]!.y;
  const end = points[points.length - 1]!.y;
  const improvement = Math.round(((start - end) / start) * 100);
  const reached = points.filter((p) => p.x <= mealsLogged).length;

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h2>Does logging actually help?</h2>
          <p className="sub">
            Yes — about {improvement}% lower error after {points[points.length - 1]!.x} meals,
            measured on people the model had never seen.
          </p>
        </div>
      </div>

      <Chart
        points={points}
        height={160}
        zeroBaseline={false}
        xLabel="meals logged"
        yLabel="mean error (mg/dL·h)"
        ariaLabel={`Error falling from ${start.toFixed(1)} to ${end.toFixed(1)} as meals are logged`}
      />

      <p className="hint">
        Most of the gain arrives by the fifth meal. You are{" "}
        {mealsLogged === 0
          ? "at the start of this curve"
          : `${reached - 1} step${reached === 2 ? "" : "s"} along it`}
        .
      </p>

      <dl className="stats">
        <div>
          <dt>No history</dt>
          <dd>{start.toFixed(1)}</dd>
        </div>
        <div>
          <dt>After {points[points.length - 1]!.x} meals</dt>
          <dd>{end.toFixed(1)}</dd>
        </div>
        <div>
          <dt>Cohort</dt>
          <dd>{meta.cohort?.n_subjects} people</dd>
        </div>
      </dl>
    </section>
  );
}
