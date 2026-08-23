/**
 * Typed client for the meal-risk API.
 *
 * Every type here is derived from `contract/openapi.json` via `schema.d.ts`,
 * which is GENERATED — never edit it by hand. If the backend changes a response
 * shape, regenerate (`npm run gen:api`) and TypeScript will point at every call
 * site that needs updating. That is the whole mechanism keeping this app and the
 * Python backend in sync; without it, drift is silent until a demo.
 *
 * The stub server and the real backend implement the same contract, so nothing
 * in this file changes when you switch from one to the other.
 */
import type { components } from "./schema";

// ---- Types re-exported with friendly names ---------------------------------

export type LabPanel = components["schemas"]["LabPanel"];
export type Meal = components["schemas"]["Meal"];
export type HistoryEntry = components["schemas"]["HistoryEntry"];
export type AssessRequest = components["schemas"]["AssessRequest"];
export type AssessResponse = components["schemas"]["AssessResponse"];
export type AlternativesRequest = components["schemas"]["AlternativesRequest"];
export type AlternativesResponse = components["schemas"]["AlternativesResponse"];
export type Meta = components["schemas"]["Meta"];
export type FieldList = components["schemas"]["FieldList"];
export type Field = components["schemas"]["Field"];
export type Edit = components["schemas"]["Edit"];
export type RiskFlag = components["schemas"]["RiskFlag"];
export type CurvePoint = components["schemas"]["CurvePoint"];
export type Confidence = components["schemas"]["Confidence"];
export type MealType = NonNullable<Meal["meal_type"]>;
export type ConfidenceBand = NonNullable<Confidence["band"]>;

/** The API's own error body. Thrown, not returned, so callers can't ignore it. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly field?: string | null,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

const BASE = "/api"; // vite proxies this to the Python backend

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // A dead backend is the single most common thing to hit while developing,
    // so say so plainly rather than surfacing "Failed to fetch".
    throw new ApiError(
      0,
      "Cannot reach the backend. Start it with:\n" +
        "  uv run --with fastapi --with uvicorn python contract/stub_server.py",
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(
      response.status,
      body?.detail ?? `${response.status} ${response.statusText}`,
      body?.field,
    );
  }
  return (await response.json()) as T;
}

// ---- Endpoints -------------------------------------------------------------

/**
 * Cohort size, thresholds, performance figures and disclaimer text.
 *
 * Call once at startup and render FROM IT. Do not hardcode 45, 0.888 or the
 * disclaimer copy into components — those numbers change when the model is
 * retrained, and a hardcoded one becomes a false claim on stage.
 */
export const getMeta = () => request<Meta>("/meta");

/**
 * The lab fields, with units, ranges and cohort medians.
 * Order your form by `importance_rank` — 1 is the strongest predictor, so a user
 * who fills in only the first three fields gets most of the available accuracy.
 */
export const getFields = () => request<FieldList>("/fields");

/** Assess one meal for one person. */
export const assess = (body: AssessRequest) =>
  request<AssessResponse>("/assess", { method: "POST", body: JSON.stringify(body) });

/** Suggested edits to a meal, smallest effective change first. */
export const alternatives = (body: AlternativesRequest) =>
  request<AlternativesResponse>("/alternatives", {
    method: "POST",
    body: JSON.stringify(body),
  });

// ---- Small helpers the UI will want ---------------------------------------

/**
 * How the risk should read in the UI.
 *
 * Deliberately returns neutral wording. The product rule is that we never call a
 * meal "dangerous" — we state a modelled probability against a named threshold,
 * because that is what a model fitted to 45 people can actually support.
 */
export function describeRisk(flag: RiskFlag, meta: Meta): string {
  const pct = Math.round((flag.probability ?? 0) * 100);
  const threshold = flag.threshold_mgdl ?? 140;
  const cohortPct = flag.cohort_percentile ?? 0;
  const n = meta.cohort?.n_subjects ?? 45;
  return (
    `${pct}% likely to exceed ${threshold} mg/dL — ` +
    `higher than ${cohortPct}% of meals from this ${n}-person cohort`
  );
}

/** Wide bands must LOOK wide. Use this to drive styling, not just a label. */
export function bandWeight(band: ConfidenceBand): { label: string; hint: string } {
  switch (band) {
    case "narrow":
      return { label: "Narrower estimate", hint: "Full lab panel and a glucose reading." };
    case "moderate":
      return { label: "Rough estimate", hint: "Some values were filled in from cohort medians." };
    default:
      return { label: "Very rough estimate", hint: "Most values were filled in. Treat as indicative only." };
  }
}
