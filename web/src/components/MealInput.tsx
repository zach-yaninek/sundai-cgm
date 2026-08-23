/**
 * Macros for one meal.
 *
 * Validation is deliberately strict. Carbohydrate in the source data runs to
 * 761 g, so the model will return a confident number for a typo'd 660 that was
 * meant to be 66. Refusing is better than answering.
 */
import { MACRO_LABELS, MACRO_UNITS, MEAL_TYPES } from "../lib/format";
import type { Meal } from "../api/client";

const RANGES: Record<string, [number, number]> = {
  carbs: [0, 800], protein: [0, 400], fat: [0, 400],
  fiber: [0, 100], calories: [0, 4000],
};

const EXAMPLES: Record<string, Meal> = {
  "Rice bowl with chicken": {
    carbs: 66, protein: 20, fat: 18, calories: 712, fiber: 4, meal_type: "dinner",
  },
  "Oats and berries": {
    carbs: 42, protein: 9, fat: 6, calories: 280, fiber: 8, meal_type: "breakfast",
  },
  "Salad with salmon": {
    carbs: 12, protein: 28, fat: 22, calories: 380, fiber: 6, meal_type: "lunch",
  },
};

export function mealErrors(meal: Meal): string[] {
  const out: string[] = [];
  for (const [key, [lo, hi]] of Object.entries(RANGES)) {
    const v = meal[key as keyof Meal] as number | undefined;
    if (v != null && (v < lo || v > hi)) {
      out.push(`${MACRO_LABELS[key]} of ${v} ${MACRO_UNITS[key]} is outside ${lo}–${hi}`);
    }
  }
  if (meal.carbs == null) out.push("Carbohydrate is required");
  if (meal.calories == null) out.push("Calories are required");
  return out;
}

export default function MealInput({
  meal, onChange, onAssess, busy,
}: {
  meal: Meal;
  onChange: (m: Meal) => void;
  onAssess: () => void;
  busy: boolean;
}) {
  const errors = mealErrors(meal);

  const setNum = (key: keyof Meal, raw: string) => {
    const next = { ...meal };
    if (raw === "") delete next[key];
    else (next[key] as number) = Number(raw);
    onChange(next);
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h2>What are you eating?</h2>
          <p className="sub">Enter the macros for the portion you will actually eat.</p>
        </div>
      </div>

      <div className="examples">
        {Object.entries(EXAMPLES).map(([name, m]) => (
          <button key={name} type="button" className="chip" onClick={() => onChange(m)}>
            {name}
          </button>
        ))}
      </div>

      <div className="field">
        <label htmlFor="meal_type">Meal</label>
        <select
          id="meal_type"
          value={meal.meal_type ?? "dinner"}
          onChange={(e) => onChange({ ...meal, meal_type: e.target.value as Meal["meal_type"] })}
        >
          {MEAL_TYPES.map((t) => (
            <option key={t} value={t}>{t[0]!.toUpperCase() + t.slice(1)}</option>
          ))}
        </select>
      </div>

      <div className="fields">
        {Object.keys(RANGES).map((key) => (
          <div className="field" key={key}>
            <label htmlFor={key}>{MACRO_LABELS[key]}</label>
            <div className="input-row">
              <input
                id={key}
                type="number"
                min={RANGES[key]![0]}
                max={RANGES[key]![1]}
                value={(meal[key as keyof Meal] as number | undefined) ?? ""}
                onChange={(e) => setNum(key as keyof Meal, e.target.value)}
              />
              <span className="unit">{MACRO_UNITS[key]}</span>
            </div>
          </div>
        ))}
      </div>

      {errors.length > 0 && (
        <ul className="errors">
          {errors.map((e) => <li key={e}>{e}</li>)}
        </ul>
      )}

      <button
        type="button"
        className="primary"
        onClick={onAssess}
        disabled={errors.length > 0 || busy}
      >
        {busy ? "Assessing…" : "Assess this meal"}
      </button>
    </section>
  );
}
