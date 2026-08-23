/**
 * Pillar 2 — would drawing bloods actually tell us anything?
 *
 * The screen answers a question a person actually asks — "is this test worth
 * it?" — so the first thing it says is an answer, in words. The measurement is
 * one click away rather than gone: the claim rests on it, and hiding it would
 * make this marketing. That is the whole shape of this component.
 *
 * AUC is not dropped, it is translated. AUC has an exact lay reading — how
 * often the model, shown one meal that crossed the threshold and one that did
 * not, picks the right one — and `describeLabValue()` restates the API's own
 * figures on that scale. No number here is invented or rounded into a claim.
 *
 * The thing this screen must never do is tell someone wearing a CGM to get a
 * glucose panel. They already have continuous glucose, which beats a single
 * fasting draw. What a draw adds is HbA1c, fasting insulin and HOMA-IR, which
 * no CGM can produce — so when there is a glucose reading on file, fasting
 * glucose sorts last and is explicitly marked as the one they already cover.
 *
 * A score of zero means a redraw would add nothing this model can use, and the
 * screen says so and offers nothing further.
 */
import { describeLabValue } from "../api/client";
import type { Field, LabValueResponse, Meta } from "../api/client";

/** The analyte a CGM already supplies, so it must not lead the ask. */
const FROM_CGM = "fasting_glu___pdl_lab";

export default function LabValue({
  data, meta, fields,
}: {
  data: LabValueResponse | null;
  meta: Meta;
  fields: Field[];
}) {
  if (!data) return null;

  const copy = describeLabValue(data);
  const threshold = meta.thresholds?.exceed_mgdl;
  const n = meta.cohort?.n_subjects;

  // Labels come from /api/fields so this screen and the lab form never end up
  // calling the same analyte two different things.
  const labelFor = (name: string): string =>
    fields.find((f) => f.name === name)?.label
    ?? name.replace(/_pdl_lab|_cal|___/g, " ").trim();

  const hasGlucoseReading = data.used_pre_meal_glucose === true;
  const missing = [...(data.missing_fields ?? [])].sort(
    (a, b) => Number(a === FROM_CGM) - Number(b === FROM_CGM),
  );
  const glucoseIsCovered = hasGlucoseReading && missing.includes(FROM_CGM);

  return (
    <section className={`card labvalue ${copy.worthwhile ? "worth" : "not-worth"}`}>
      <div className="card-head">
        <div>
          <h2>Would a blood test help?</h2>
          <p className="sub">
            About how well this model can predict your meals — not a screening
            result, and not a reason to test for anything.
          </p>
        </div>
      </div>

      <p className="verdict">{copy.verdict}</p>
      <p className="risk-sentence">{copy.plain}</p>

      <p className="plain-accuracy">
        Shown one meal that took someone past {threshold} mg/dL and one that did
        not, this model picks the right one about{" "}
        <strong>{copy.nowInHundred} times in 100</strong> with what it knows
        about you today
        {copy.worthwhile ? (
          <>
            , and about <strong>{copy.afterInHundred} times in 100</strong> after
            the draw.
          </>
        ) : (
          "."
        )}
      </p>

      {copy.worthwhile && (
        <div className="panel-ask">
          <h3>What to ask for</h3>
          <p className="panel-name">{data.recommended_panel}</p>
          <ul className="analytes">
            {missing.map((f) => (
              <li key={f} className={f === FROM_CGM && hasGlucoseReading ? "covered" : undefined}>
                {labelFor(f)}
              </li>
            ))}
          </ul>
          {glucoseIsCovered && (
            <p className="hint">
              You already have a glucose reading, so that one is the least of it
              — a fasting glucose comes with the draw anyway. The values a
              glucose monitor cannot give you are the ones doing the work here.
            </p>
          )}
        </div>
      )}

      <details className="more">
        <summary>How this was measured</summary>
        <div className="more-body">
          <p>
            "Times in 100" above is AUC, restated. AUC is the chance the model
            ranks a meal that crossed the threshold above one that did not: 50 in
            100 is a coin flip, 100 in 100 would be perfect. Every figure below
            was measured under cold-start cross-validation — held-out people the
            model had never seen{n ? `, across ${n} adults in one study` : ""}.
          </p>

          <dl className="stats">
            <div>
              <dt>Flag accuracy now</dt>
              <dd>{data.auc_now?.toFixed(3)}</dd>
            </div>
            {copy.worthwhile && (
              <>
                <div>
                  <dt>After the draw</dt>
                  <dd>{data.auc_after_draw?.toFixed(3)}</dd>
                </div>
                <div>
                  <dt>Gain</dt>
                  <dd>+{((data.auc_after_draw ?? 0) - (data.auc_now ?? 0)).toFixed(3)} AUC</dd>
                </div>
              </>
            )}
            <div>
              <dt>Panel on file</dt>
              <dd>{data.current_tier}</dd>
            </div>
          </dl>

          <p className="hint">{data.reason}</p>

          {copy.worthwhile && data.recommended_tier === "core" && (
            <p className="hint">
              A full lipid panel adds almost nothing on top of these three, and
              does worse than them when there is no glucose reading — more
              features fitted against the same small cohort. So this asks for the
              three that carry the signal rather than everything a lab can run.
            </p>
          )}

          <p className="provenance">Model version {data.model_version}</p>
        </div>
      </details>
    </section>
  );
}
