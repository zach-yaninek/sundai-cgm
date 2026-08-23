/**
 * The other half of the privacy claim.
 *
 * The app tells people their labs and meals never leave the browser. That is
 * true, and on its own it is only half an answer: data that never leaves is
 * still data that is sitting there. Until this existed, `clearEverything()` was
 * written, exported, and called by nothing — the promise was made in the footer
 * and could only be kept from devtools.
 *
 * Two clicks rather than one, and no `window.confirm`: this wipes a lab panel
 * someone typed in by hand and a history they built up over weeks, so a
 * mis-click should not be able to do it, and a dialog the browser can suppress
 * is not a safeguard.
 */
import { useState } from "react";

export default function DeleteData({ onDelete }: { onDelete: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [done, setDone] = useState(false);

  if (done) {
    return (
      <p className="delete-done" role="status">
        Deleted. Your labs, your meal history and your consent are gone from this
        browser.
      </p>
    );
  }

  if (!confirming) {
    return (
      <button type="button" className="link danger" onClick={() => setConfirming(true)}>
        Delete my data
      </button>
    );
  }

  return (
    <span className="delete-confirm">
      <span>Delete your labs, meal history and consent from this browser?</span>
      <button
        type="button"
        className="link danger"
        onClick={() => {
          onDelete();
          setDone(true);
        }}
      >
        Yes, delete
      </button>
      <button type="button" className="link" onClick={() => setConfirming(false)}>
        Cancel
      </button>
    </span>
  );
}
