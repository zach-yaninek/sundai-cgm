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
export type LabValueResponse = components["schemas"]["LabValueResponse"];
export type ExplainResponse = components["schemas"]["ExplainResponse"];
export type Driver = components["schemas"]["Driver"];
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

// In dev, Vite proxies "/api" to the local backend. In a static build there is
// no proxy, so the deployed frontend needs the API's absolute origin.
const BASE = import.meta.env.VITE_API_BASE
  ? `${String(import.meta.env.VITE_API_BASE).replace(/\/$/, "")}/api`
  : "/api";

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

/** Would drawing this person's bloods sharpen what we can tell them? */
export const labValue = (body: { labs: LabPanel; pre_meal_glucose?: number | null }) =>
  request<LabValueResponse>("/lab-value", { method: "POST", body: JSON.stringify(body) });

/**
 * Narration of one assessment.
 *
 * Always resolves — the backend falls back to a deterministic template when
 * there is no API key or a response fails validation. Check `source` and show
 * it; hiding which one produced the text would misrepresent the system.
 */
export const explain = (body: AssessRequest) =>
  request<ExplainResponse>("/explain", { method: "POST", body: JSON.stringify(body) });

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

/**
 * The lab-value screen in plain language.
 *
 * The API speaks in AUC because that is what was measured. AUC has an exact
 * lay reading, though — it is how often the model, shown one meal that crossed
 * the threshold and one that did not, picks the right one. So the lead copy
 * states that, and the AUC figures themselves move behind the disclosure.
 *
 * Nothing here is a number this file invented: `nowInHundred` and
 * `afterInHundred` are the API's own AUCs, restated on a scale people read
 * without training.
 */
export interface LabValueCopy {
  /** Short answer to the question the tab asks. */
  verdict: string;
  /** Why, without jargon. */
  plain: string;
  /** AUC now and after the draw, as "picks the right one N times in 100". */
  nowInHundred: number;
  afterInHundred: number;
  /** Whether a draw is worth recommending at all. */
  worthwhile: boolean;
}

/** Small counts read better spelled out mid-sentence, and never open one as a digit. */
const COUNT_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven"];
const spell = (n: number): string => COUNT_WORDS[n] ?? String(n);

export function describeLabValue(data: LabValueResponse): LabValueCopy {
  const nowInHundred = Math.round((data.auc_now ?? 0) * 100);
  const afterInHundred = Math.round((data.auc_after_draw ?? 0) * 100);
  const missing = data.missing_fields ?? [];
  // A draw has to change something the model can act on. Below this it is
  // measurement noise on 45 subjects, and saying "yes" would be selling a test.
  const worthwhile = (data.score ?? 0) > 0.05 && missing.length > 0;

  if (worthwhile) {
    const one = missing.length === 1;
    const count = spell(missing.length);
    return {
      verdict: "Yes — a blood test would sharpen this",
      plain:
        `${count.charAt(0).toUpperCase()}${count.slice(1)} of the values this ` +
        `model leans on hardest ${one ? "is" : "are"} not on file, so it is ` +
        `standing in cohort averages instead. One fasting blood draw would ` +
        `replace ${one ? "that stand-in" : "those stand-ins"} with your own numbers.`,
      nowInHundred,
      afterInHundred,
      worthwhile,
    };
  }

  // Score at or near zero, but the reason differs: either everything useful is
  // already known, or something is missing and would not move the needle. Those
  // are different sentences and conflating them overstates the first.
  if (missing.length > 0) {
    const one = missing.length === 1;
    return {
      verdict: "Not worth drawing again",
      plain:
        `There ${one ? "is" : "are"} still ${spell(missing.length)} value` +
        `${one ? "" : "s"} this model can use that ${one ? "is" : "are"} not on ` +
        `file, but filling ${one ? "it" : "them"} in would not measurably change ` +
        `what it can tell you about your meals.`,
      nowInHundred,
      afterInHundred,
      worthwhile,
    };
  }

  return {
    verdict: "No — nothing further to draw",
    plain:
      "Every value this model can use is already on file. Drawing again would " +
      "not improve what it can tell you about your meals.",
    nowInHundred,
    afterInHundred,
    worthwhile,
  };
}
