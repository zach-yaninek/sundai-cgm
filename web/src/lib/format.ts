/** Number and unit formatting, in one place so the UI reads consistently. */

export const pct = (p: number | undefined | null): string =>
  p == null ? "—" : `${Math.round(p * 100)}%`;

export const mgdl = (v: number | undefined | null): string =>
  v == null ? "—" : `${Math.round(v)} mg/dL`;

export const iauc = (v: number | undefined | null): string =>
  v == null ? "—" : `${Math.round(v)} mg/dL·h`;

export const grams = (v: number | undefined | null): string =>
  v == null ? "—" : `${Math.round(v)} g`;

export const signed = (v: number | undefined | null, unit = ""): string =>
  v == null ? "—" : `${v > 0 ? "+" : ""}${Math.round(v * 10) / 10}${unit}`;

/**
 * Risk tier from the calibrated probability.
 *
 * Drives colour and weight, never wording — the sentence always comes from
 * `describeRisk()` so the phrasing rule lives in one place.
 */
export type RiskTier = "low" | "moderate" | "high";

export const riskTier = (p: number | undefined | null): RiskTier =>
  p == null ? "moderate" : p >= 0.66 ? "high" : p >= 0.34 ? "moderate" : "low";

export const MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack"] as const;

/** Lab field labels are supplied by /api/fields; this is only for meal macros. */
export const MACRO_LABELS: Record<string, string> = {
  carbs: "Carbohydrate",
  protein: "Protein",
  fat: "Fat",
  fiber: "Fibre",
  calories: "Calories",
};

export const MACRO_UNITS: Record<string, string> = {
  carbs: "g", protein: "g", fat: "g", fiber: "g", calories: "kcal",
};
