/**
 * Lab panel + consent gate.
 *
 * Fields come from `/api/fields` via App, already ordered by `importance_rank`,
 * so someone who fills only the first three gets most of the accuracy that is
 * available. Every field is optional; blanks are filled from cohort medians and
 * reported back.
 */
import { useState } from "react";
import type { Field, LabPanel, Meta } from "../api/client";
import DemoPersona from "./DemoPersona";
import type { Persona } from "./DemoPersona";

interface Props {
  meta: Meta;
  /** From `/api/fields`, already ordered by `importance_rank`. Owned by App. */
  fields: Field[];
  labs: LabPanel;
  preMealGlucose: number | null;
  onChange: (labs: LabPanel, preMealGlucose: number | null) => void;
  consented: boolean;
  onConsent: () => void;
}

export default function OnboardLabs({
  meta, fields, labs, preMealGlucose, onChange, consented, onConsent,
}: Props) {
  const [excluded, setExcluded] = useState<Record<string, boolean>>({});
  const [accepted, setAccepted] = useState(false);

  const anyExclusion = Object.values(excluded).some(Boolean);
  const filled = fields.filter((f) => labs[f.name as keyof LabPanel] != null).length;

  const setField = (name: string, raw: string) => {
    const next = { ...labs };
    if (raw === "") delete next[name as keyof LabPanel];
    else next[name as keyof LabPanel] = Number(raw) as never;
    onChange(next, preMealGlucose);
  };

  const usePersona = (p: Persona) => onChange(p.labs, p.preMealGlucose);

  if (!consented) {
    return (
      <section className="card gate">
        <h2>Before you start</h2>
        <p className="disclaimer">{meta.disclaimer?.text}</p>

        <h3>This is not for you if any of these apply</h3>
        <ul className="exclusions">
          {(meta.exclusions ?? []).map((e) => (
            <li key={e}>
              <label>
                <input
                  type="checkbox"
                  checked={!!excluded[e]}
                  onChange={(ev) => setExcluded({ ...excluded, [e]: ev.target.checked })}
                />
                <span>{e}</span>
              </label>
            </li>
          ))}
        </ul>

        {anyExclusion && (
          <p className="blocked">
            This tool is not appropriate in that case, and it will not produce an
            estimate. Please talk to a clinician instead.
          </p>
        )}

        <label className="accept">
          <input
            type="checkbox"
            checked={accepted}
            onChange={(e) => setAccepted(e.target.checked)}
            disabled={anyExclusion}
          />
          <span>I understand this is a research demo, not a medical device.</span>
        </label>

        <button
          type="button"
          className="primary"
          disabled={!accepted || anyExclusion}
          onClick={onConsent}
        >
          Continue
        </button>
      </section>
    );
  }

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h2>Your bloodwork</h2>
          <p className="sub">
            All optional. {filled} of {fields.length} filled — anything blank is
            estimated from cohort medians, and the app tells you which.
          </p>
        </div>
      </div>

      <DemoPersona onPick={usePersona} />

      <div className="field glucose">
        <label htmlFor="pmg">
          Current glucose reading <span className="opt">optional, from a CGM or fingerstick</span>
        </label>
        <div className="input-row">
          <input
            id="pmg"
            type="number"
            value={preMealGlucose ?? ""}
            min={40}
            max={400}
            onChange={(e) =>
              onChange(labs, e.target.value === "" ? null : Number(e.target.value))
            }
          />
          <span className="unit">mg/dL</span>
        </div>
        <p className="hint">
          Supplying this moves the flag from AUC{" "}
          {meta.performance?.auc_without_glucose} to {meta.performance?.auc_with_glucose}.
        </p>
      </div>

      <div className="fields">
        {fields.map((f, i) => (
          <div className="field" key={f.name}>
            <label htmlFor={f.name}>
              {f.label}
              {i < 3 && <span className="rank">strongest</span>}
            </label>
            <div className="input-row">
              <input
                id={f.name}
                type="number"
                min={f.min}
                max={f.max}
                placeholder={f.cohort_median != null ? String(f.cohort_median) : ""}
                value={(labs[f.name as keyof LabPanel] as number | undefined) ?? ""}
                onChange={(e) => setField(f.name, e.target.value)}
              />
              <span className="unit">{f.unit}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
