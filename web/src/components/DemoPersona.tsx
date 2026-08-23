/**
 * One-click lab panels, so nobody types fourteen values on stage.
 *
 * These are INVENTED, clinically plausible values — not copied from any study
 * participant. The project does not redistribute the source data, and that has
 * to include the demo.
 */
import type { LabPanel } from "../api/client";

export interface Persona {
  key: string;
  name: string;
  blurb: string;
  labs: LabPanel;
  preMealGlucose: number;
}

export const PERSONAS: Persona[] = [
  {
    key: "healthy",
    name: "Metabolically healthy",
    blurb: "HOMA-IR 0.9 · normal HbA1c",
    preMealGlucose: 92,
    labs: {
      age: 30, bmi: 22.0, body_weight: 140, height: 67,
      a1c_pdl_lab: 5.2, fasting_glu___pdl_lab: 88, insulin: 4.0,
      triglycerides: 70, cholesterol: 175, hdl: 65, non_hdl: 110,
      ldl_cal: 95, vldl_cal: 14, cho_hdl_ratio: 2.7,
    },
  },
  {
    key: "resistant",
    name: "Insulin resistant",
    blurb: "HOMA-IR 5.1 · HbA1c 6.2%",
    preMealGlucose: 108,
    labs: {
      age: 55, bmi: 31.0, body_weight: 200, height: 68,
      a1c_pdl_lab: 6.2, fasting_glu___pdl_lab: 115, insulin: 18.0,
      triglycerides: 190, cholesterol: 210, hdl: 38, non_hdl: 172,
      ldl_cal: 130, vldl_cal: 38, cho_hdl_ratio: 5.5,
    },
  },
  {
    key: "unknown",
    name: "No bloodwork",
    blurb: "Demographics only — see what a draw would add",
    preMealGlucose: 98,
    labs: { age: 42, bmi: 27.0 },
  },
];

export default function DemoPersona({
  onPick,
}: {
  onPick: (p: Persona) => void;
}) {
  return (
    <div className="personas">
      <p className="personas-note">
        Example panels for the demo. Invented values, not real participants.
      </p>
      <div className="persona-row">
        {PERSONAS.map((p) => (
          <button key={p.key} type="button" className="persona" onClick={() => onPick(p)}>
            <strong>{p.name}</strong>
            <span>{p.blurb}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
