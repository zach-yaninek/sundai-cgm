/** Macros, meal type, optional pre-meal glucose.
 *
 * Validate hard: carbohydrate in the source data runs 0-761 g, and a typo'd 660
 * for 66 must not sail through to a confident prediction. Ranges are in
 * `getFields()` for labs and in the OpenAPI `Meal` schema for macros.
 *
 * Prompt for pre-meal glucose — optional, but it lifts flag AUC 0.841 -> 0.888. */
export default function MealInput() {
  return <div>TODO: MealInput</div>;
}
