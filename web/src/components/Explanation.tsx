/**
 * Why the model said that.
 *
 * Loaded lazily, after the numbers are already on screen, because the narration
 * may involve an external call and the assessment must never wait on one.
 *
 * `source` is shown, not hidden. "template" means the deterministic fallback ran
 * — no API key, an upstream failure, or a response that failed validation — and
 * a reader deserves to know which produced the words they are reading.
 *
 * That fallback is labelled "standard wording", not "generated locally", which
 * is what it used to say. "Locally" reads as *a local model wrote this* when it
 * means *no model wrote this*: the sentence is fixed text filled in with this
 * model's own attributions, and nothing has to be installed for it to work.
 * A label that makes a reader wonder what they are missing is doing the
 * opposite of disclosure.
 */
import type { ExplainResponse } from "../api/client";

export default function Explanation({
  data, loading,
}: {
  data: ExplainResponse | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <section className="card explain">
        <h3>Why</h3>
        <p className="muted">Working out what drove this…</p>
      </section>
    );
  }
  if (!data) return null;

  const drivers = data.drivers_used ?? [];
  const max = Math.max(...drivers.map((d) => d.contribution ?? 0), 0.001);

  return (
    <section className="card explain">
      <div className="card-head">
        <h3>Why</h3>
        <span className={`source source-${data.source}`}>
          {data.source === "claude" ? "written by Claude" : "standard wording"}
        </span>
      </div>

      <p className="explain-headline">{data.headline}</p>

      {drivers.length > 0 && (
        <ul className="drivers">
          {drivers.map((d) => (
            <li key={d.feature}>
              <span className="driver-label">{d.label}</span>
              <span className={`driver-bar driver-${d.direction}`}>
                <i style={{ width: `${((d.contribution ?? 0) / max) * 100}%` }} />
              </span>
              <span className="driver-dir">{d.direction}</span>
            </li>
          ))}
        </ul>
      )}

      <p className="explain-caveat">{data.caveat}</p>

      {data.rejected_reason && (
        <p className="hint">
          A written explanation was discarded ({data.rejected_reason}) and the
          standard wording was used instead.
        </p>
      )}
    </section>
  );
}
