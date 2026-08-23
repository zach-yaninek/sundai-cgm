/**
 * Everything personal lives here, in the browser, and nowhere else.
 *
 * The backend is stateless by design: the lab panel and the meal history are
 * sent on each request and never stored server-side. That is the app's privacy
 * story, and it is only true if this stays the single source of that data.
 *
 * Do not add a server-side session, a user table, or an analytics call carrying
 * any of these values.
 */
import type { HistoryEntry, LabPanel, Meal } from "../api/client";

const KEYS = {
  labs: "cgm.labs.v1",
  history: "cgm.history.v1",
  consent: "cgm.consent.v1",
} as const;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    // Corrupt or unavailable storage must not white-screen the app.
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private browsing, quota — degrade to in-memory for the session */
  }
}

// ---- Lab panel -------------------------------------------------------------

export const loadLabs = (): LabPanel => read<LabPanel>(KEYS.labs, {});
export const saveLabs = (labs: LabPanel): void => write(KEYS.labs, labs);

// ---- Meal history ----------------------------------------------------------

export const loadHistory = (): HistoryEntry[] => read<HistoryEntry[]>(KEYS.history, []);

/**
 * Append one logged meal.
 *
 * `observed_peak` / `observed_iauc` are what ACTUALLY happened — a later reading
 * the user enters, not the prediction. Storing a prediction here would teach the
 * personalisation layer from its own output, and the offset would drift away
 * from reality while looking like it was learning.
 */
export function addHistory(
  meal: Meal,
  observed: { peak?: number; iauc?: number; preMealGlucose?: number },
): HistoryEntry[] {
  const entry: HistoryEntry = {
    meal,
    pre_meal_glucose: observed.preMealGlucose ?? null,
    observed_peak: observed.peak ?? null,
    observed_iauc: observed.iauc ?? null,
    logged_at: new Date().toISOString(),
  };
  const next = [...loadHistory(), entry];
  write(KEYS.history, next);
  return next;
}

/** Only entries with a real observed outcome can teach the model anything. */
export const usableHistory = (): HistoryEntry[] =>
  loadHistory().filter((h) => h.observed_peak != null || h.observed_iauc != null);

export function clearHistory(): void {
  write(KEYS.history, []);
}

/** Full wipe, for the "delete my data" control. Please build one. */
export function clearEverything(): void {
  Object.values(KEYS).forEach((k) => {
    try {
      localStorage.removeItem(k);
    } catch {
      /* ignore */
    }
  });
}

// ---- Consent ---------------------------------------------------------------

export interface Consent {
  disclaimerId: string;
  acceptedAt: string;
  exclusionsConfirmed: boolean;
}

export const loadConsent = (): Consent | null => read<Consent | null>(KEYS.consent, null);

export const saveConsent = (disclaimerId: string): void =>
  write(KEYS.consent, {
    disclaimerId,
    acceptedAt: new Date().toISOString(),
    exclusionsConfirmed: true,
  } satisfies Consent);

/**
 * Consent is tied to a specific disclaimer version. If the backend ships new
 * disclaimer text, the user has not agreed to it yet and must be asked again.
 */
export const hasConsented = (currentDisclaimerId: string): boolean =>
  loadConsent()?.disclaimerId === currentDisclaimerId;
