/** Logged meals and what actually happened.
 *
 * `src/lib/storage.ts` handles persistence. Record the OBSERVED outcome the user
 * enters later — never the prediction. Feeding predictions back would teach the
 * personalisation layer from its own output.
 *
 * Show `personalization.meals_logged` climbing; that is the "it learns you" story. */
export default function History() {
  return <div>TODO: History</div>;
}
