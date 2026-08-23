/**
 * The result.
 *
 * Two rules are load-bearing here. The sentence comes from `describeRisk()` so
 * the phrasing lives in one place and never drifts into "dangerous". And the
 * confidence band changes how this *looks* — a wide band renders lighter, with a
 * dashed curve — because a rough estimate shown in confident type is a lie the
 * API went out of its way not to tell.
 */
import { bandWeight, describeRisk } from "../api/client";
import type { AssessResponse, Meta } from "../api/client";
import { iauc, mgdl, pct, riskTier } from "../lib/format";
import Chart from "./Chart";

export default function RiskCard({
  result, meta,
}: {
  result: AssessResponse;
  meta: Meta;
}) {
  const flag = result.exceeds_140!;
  const tier = riskTier(flag.probability);
  const band = result.confidence?.band ?? "moderate";
  const weight = bandWeight(band);
  const imputed = result.confidence?.imputed_fields ?? [];

  const curve = (result.curve ?? []).map((p) => ({ x: p.minute ?? 0, y: p.delta ?? 0 }));
  const baseline = (result.predicted_peak_mgdl ?? 0) - Math.max(...curve.map((c) => c.y), 0);

  return (
    <section className={`card risk risk-${tier} band-${band}`}>
      <div className="risk-top">
        <div className="risk-figure">
          <strong>{pct(flag.probability)}</strong>
          <span>chance of exceeding {flag.threshold_mgdl} mg/dL</span>
        </div>
        <div className={`band band-tag-${band}`}>
          <strong>{weight.label}</strong>
          <span>{weight.hint}</span>
        </div>
      </div>

      <p className="risk-sentence">{describeRisk(flag, meta)}</p>

      <dl className="stats">
        <div>
          <dt>Predicted peak</dt>
          <dd>{mgdl(result.predicted_peak_mgdl)}</dd>
        </div>
        <div>
          <dt>Predicted rise</dt>
          <dd>{iauc(result.predicted_iauc)}</dd>
        </div>
        <div>
          <dt>Meals you've logged</dt>
          <dd>{result.personalization?.meals_logged ?? 0}</dd>
        </div>
        <div>
          <dt>Personal adjustment</dt>
          <dd>{result.personalization?.offset_applied ? `${result.personalization.offset_applied > 0 ? "+" : ""}${result.personalization.offset_applied}` : "none yet"}</dd>
        </div>
      </dl>

      {curve.length > 1 && (
        <div className="curve">
          <Chart
            points={curve}
            uncertain={band === "wide"}
            xLabel="minutes after eating"
            yLabel="mg/dL above baseline"
            threshold={{
              y: (flag.threshold_mgdl ?? 140) - baseline,
              label: `${flag.threshold_mgdl} mg/dL`,
            }}
            ariaLabel={`Predicted glucose curve peaking at ${Math.round(result.predicted_peak_mgdl ?? 0)} mg/dL`}
          />
          <p className="hint">{result.curve_note}</p>
        </div>
      )}

      {result.personalization?.learned_slope && (
        <p className="hint">
          This adjustment is for this meal, not a fixed amount. With enough
          logged meals the model has learned how your response scales, so it
          corrects a large meal differently from a small one.
        </p>
      )}

      {imputed.length > 0 && (
        <p className="imputed">
          {imputed.length} input{imputed.length === 1 ? " was" : "s were"} estimated
          from cohort medians. Filling them in narrows this.
        </p>
      )}

      <p className="provenance">
        Model fitted to {meta.cohort?.n_subjects} adults in one study
        {" · "}{meta.cohort?.age_range}
        {" · "}version {result.model_version}
      </p>
    </section>
  );
}
