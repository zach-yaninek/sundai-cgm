/**
 * Pillar 2 — would drawing bloods actually tell us anything?
 *
 * The thing this screen must never do is tell someone wearing a CGM to get a
 * glucose panel. They already have continuous glucose, which beats a single
 * fasting draw. What a draw adds is HbA1c, fasting insulin and HOMA-IR, which no
 * CGM can produce — and the copy leads with those analytes for that reason.
 *
 * A score of zero means a redraw would add nothing this model can use, and the
 * screen says so and offers nothing further.
 */
import type { LabValueResponse } from "../api/client";
import { pct } from "../lib/format";

export default function LabValue({ data }: { data: LabValueResponse | null }) {
  if (!data) return null;

  const worthwhile = (data.score ?? 0) > 0.05;
  const gain = ((data.auc_after_draw ?? 0) - (data.auc_now ?? 0));

  return (
    <section className={`card labvalue ${worthwhile ? "worth" : "not-worth"}`}>
      <div className="card-head">
        <div>
          <h2>Would a blood test help?</h2>
          <p className="sub">
            Measured as information gain, not as a screening result.
          </p>
        </div>
      </div>

      <div className="risk-top">
        <div className="risk-figure">
          <strong>{pct(data.score)}</strong>
          <span>of the available gain is still on the table</span>
        </div>
      </div>

      <p className="risk-sentence">{data.reason}</p>

      {worthwhile ? (
        <>
          <div className="panel-ask">
            <h3>What to ask for</h3>
            <p className="panel-name">{data.recommended_panel}</p>
            <ul className="analytes">
              {(data.missing_fields ?? []).map((f) => (
                <li key={f}>{f.replace(/_pdl_lab|_cal|___/g, " ").trim()}</li>
              ))}
            </ul>
            <p className="hint">
              A full lipid panel adds roughly 0.002 AUC on top of these, and
              performs worse than them when there is no glucose reading. This asks
              for the three that carry the signal.
            </p>
          </div>

          <dl className="stats">
            <div>
              <dt>Flag accuracy now</dt>
              <dd>{data.auc_now?.toFixed(3)}</dd>
            </div>
            <div>
              <dt>After the draw</dt>
              <dd>{data.auc_after_draw?.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Gain</dt>
              <dd>+{gain.toFixed(3)} AUC</dd>
            </div>
          </dl>
        </>
      ) : (
        <p className="empty">
          Nothing further to draw. The panel on file already covers everything
          this model can use.
        </p>
      )}
    </section>
  );
}
