"use client";

/**
 * Cookie Consent Banner — Item #5
 * GDPR-compliant. Stores preference in localStorage.
 * Google Analytics only loads after explicit consent (Item #19).
 */
import { useEffect, useState } from "react";

const STORAGE_KEY = "cookie_consent";

export function CookieBanner() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Only show if no decision has been made yet
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) setVisible(true);
  }, []);

  const handleAccept = () => {
    localStorage.setItem(STORAGE_KEY, "accepted");
    setVisible(false);
    // Enable GA4 by dispatching a custom event the AnalyticsScript listens for
    window.dispatchEvent(new CustomEvent("cookie_consent_accepted"));
  };

  const handleReject = () => {
    localStorage.setItem(STORAGE_KEY, "rejected");
    setVisible(false);
  };

  if (!visible) return null;

  return (
    <div
      className="cookie-banner"
      role="dialog"
      aria-modal="true"
      aria-labelledby="cookie-title"
      aria-describedby="cookie-desc"
    >
      <h3 id="cookie-title" style={{ color: "var(--color-text)", marginBottom: 8, fontSize: "1rem" }}>
        🍪 Cookie Preferences
      </h3>
      <p id="cookie-desc">
        We use cookies to understand how visitors interact with this research tool (Google Analytics).
        No personal data is stored or sold. See our{" "}
        <a href="/privacy">Privacy Policy</a>.
      </p>
      <div className="cookie-actions">
        <button
          className="btn btn-primary"
          onClick={handleAccept}
          aria-label="Accept all cookies including analytics"
          style={{ fontSize: "0.88rem", padding: "10px 20px" }}
        >
          Accept All
        </button>
        <button
          className="btn btn-secondary"
          onClick={handleReject}
          aria-label="Reject non-essential cookies"
          style={{ fontSize: "0.88rem", padding: "10px 20px" }}
        >
          Reject Non-Essential
        </button>
      </div>
    </div>
  );
}
