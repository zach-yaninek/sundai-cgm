/** Suggested edits, smallest effective change first.
 *
 * Each edit carries `delta_probability` (negative = lower risk) and a
 * `resulting_meal`. `from_your_history[]` holds the user's OWN past low-response
 * meals of this type and is empty until they have logged some.
 *
 * An empty `edits` array means the meal is already in their lower range. Say so.
 * Do not invent a suggestion to fill the space.
 *
 * Copy rule: "changes this model predicts would lower your response", never
 * "eat this instead". This is the sharpest edge in the product. */
export default function Alternatives() {
  return <div>TODO: Alternatives</div>;
}
