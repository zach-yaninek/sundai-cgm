/**
 * Consent gate, then three tabs — one per pillar.
 *
 * No router: four destinations do not justify the dependency or the bundle.
 *
 * The one sequencing decision worth knowing: assessment and narration are
 * separate requests. The numbers render as soon as the model returns, and the
 * explanation fills in afterwards, so an external call can never delay or block
 * the thing the user actually asked for.
 */
import { useCallback, useEffect, useState } from "react";
import {
  alternatives as fetchAlternatives,
  ApiError,
  assess,
  explain as fetchExplain,
  getFields,
  getMeta,
  labValue as fetchLabValue,
} from "./api/client";
import type {
  AlternativesResponse,
  AssessResponse,
  ExplainResponse,
  Field,
  HistoryEntry,
  LabPanel,
  LabValueResponse,
  Meal,
  Meta,
} from "./api/client";
import Alternatives from "./components/Alternatives";
import Explanation from "./components/Explanation";
import History from "./components/History";
import LabValue from "./components/LabValue";
import LearningCurve from "./components/LearningCurve";
import MealInput from "./components/MealInput";
import OnboardLabs from "./components/OnboardLabs";
import RiskCard from "./components/RiskCard";
import {
  hasConsented,
  loadHistory,
  loadLabs,
  saveConsent,
  saveLabs,
  usableHistory,
} from "./lib/storage";

type Tab = "meal" | "bloods" | "learning";

const DEFAULT_MEAL: Meal = {
  carbs: 66, protein: 20, fat: 18, fiber: 4, calories: 712, meal_type: "dinner",
};

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  // Fetched once here rather than inside each component that needs it: the lab
  // form and the blood-test screen both label fields from it, and two copies of
  // the same list is how the two screens start naming the same analyte
  // differently.
  const [fields, setFields] = useState<Field[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [consented, setConsented] = useState(false);

  const [labs, setLabs] = useState<LabPanel>(loadLabs);
  const [preMealGlucose, setPreMealGlucose] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);

  const [tab, setTab] = useState<Tab>("meal");
  const [meal, setMeal] = useState<Meal>(DEFAULT_MEAL);
  const [result, setResult] = useState<AssessResponse | null>(null);
  const [alts, setAlts] = useState<AlternativesResponse | null>(null);
  const [narration, setNarration] = useState<ExplainResponse | null>(null);
  const [narrating, setNarrating] = useState(false);
  const [bloods, setBloods] = useState<LabValueResponse | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getMeta()
      .then((m) => {
        setMeta(m);
        setConsented(hasConsented(m.disclaimer?.id ?? "v1"));
      })
      .catch((e: unknown) =>
        setError(e instanceof ApiError ? e.message : String(e)),
      );
  }, []);

  useEffect(() => {
    getFields()
      .then((f) =>
        setFields([...f.fields].sort(
          (a, b) => (a.importance_rank ?? 99) - (b.importance_rank ?? 99),
        )),
      )
      .catch(() => setFields([]));
  }, []);

  const updateLabs = useCallback((next: LabPanel, glucose: number | null) => {
    setLabs(next);
    setPreMealGlucose(glucose);
    saveLabs(next);
    // Any change to the panel invalidates what is on screen.
    setResult(null);
    setAlts(null);
    setNarration(null);
    setBloods(null);
  }, []);

  const runAssessment = useCallback(async () => {
    setBusy(true);
    setError(null);
    const body = {
      labs,
      meal,
      pre_meal_glucose: preMealGlucose,
      history: usableHistory(),
    };
    try {
      const assessment = await assess(body);
      setResult(assessment);
      setNarration(null);

      // Fire-and-forget: the numbers are already rendered.
      setNarrating(true);
      fetchExplain(body)
        .then(setNarration)
        .catch(() => setNarration(null))
        .finally(() => setNarrating(false));

      fetchAlternatives(body).then(setAlts).catch(() => setAlts(null));
    } catch (e: unknown) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [labs, meal, preMealGlucose, history]);

  useEffect(() => {
    if (tab !== "bloods" || !consented) return;
    fetchLabValue({ labs, pre_meal_glucose: preMealGlucose })
      .then(setBloods)
      .catch(() => setBloods(null));
  }, [tab, labs, preMealGlucose, consented]);

  if (error && !meta) {
    return (
      <main className="wrap">
        <h1>Backend unreachable</h1>
        <pre className="errorbox">{error}</pre>
      </main>
    );
  }
  if (!meta) {
    return <main className="wrap"><p className="muted">Loading…</p></main>;
  }

  const logged = usableHistory().length;

  return (
    <>
      <header className="topbar">
        <div className="wrap topbar-inner">
          <div className="brand">
            <span className="mark" aria-hidden="true" />
            <span>Meal response</span>
          </div>
          <span className="cohort-tag">
            {meta.cohort?.n_subjects} people · research demo
          </span>
        </div>
      </header>

      <main className="wrap">
        {!consented ? (
          <OnboardLabs
            meta={meta}
            fields={fields}
            labs={labs}
            preMealGlucose={preMealGlucose}
            onChange={updateLabs}
            consented={false}
            onConsent={() => {
              saveConsent(meta.disclaimer?.id ?? "v1");
              setConsented(true);
            }}
          />
        ) : (
          <>
            <nav className="tabs" role="tablist">
              {([
                ["meal", "Assess a meal"],
                ["bloods", "Would a test help?"],
                ["learning", `Learning${logged ? ` (${logged})` : ""}`],
              ] as [Tab, string][]).map(([key, label]) => (
                <button
                  key={key}
                  role="tab"
                  aria-selected={tab === key}
                  className={tab === key ? "tab active" : "tab"}
                  onClick={() => setTab(key)}
                >
                  {label}
                </button>
              ))}
            </nav>

            {error && <p className="errorbox">{error}</p>}

            {tab === "meal" && (
              <>
                <OnboardLabs
                  meta={meta}
                  fields={fields}
                  labs={labs}
                  preMealGlucose={preMealGlucose}
                  onChange={updateLabs}
                  consented
                  onConsent={() => undefined}
                />
                <MealInput meal={meal} onChange={setMeal} onAssess={runAssessment} busy={busy} />
                {result && <RiskCard result={result} meta={meta} />}
                {result && <Explanation data={narration} loading={narrating} />}
                {result && <Alternatives data={alts} onApply={setMeal} />}
              </>
            )}

            {tab === "bloods" && (
              <LabValue data={bloods} meta={meta} fields={fields} />
            )}

            {tab === "learning" && (
              <>
                <LearningCurve meta={meta} mealsLogged={logged} />
                <History history={history} lastMeal={result ? meal : null} onChange={setHistory} />
              </>
            )}
          </>
        )}

        <footer className="foot">
          <p>{meta.disclaimer?.text}</p>
          <p>
            {meta.cohort?.citation} Nothing you enter leaves your browser except
            to be scored; no personal data is stored on the server.
          </p>
        </footer>
      </main>
    </>
  );
}
