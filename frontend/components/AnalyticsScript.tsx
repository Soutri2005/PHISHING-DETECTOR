"use client";

/**
 * Analytics Script — Item #19
 * Loads Google Analytics 4 only after cookie consent is granted.
 * Uses NEXT_PUBLIC_GA_ID env var — never hardcoded (Item #3).
 */
import { useEffect } from "react";
import Script from "next/script";
import { useState } from "react";

const GA_ID = process.env.NEXT_PUBLIC_GA_ID;

export function AnalyticsScript() {
  const [consented, setConsented] = useState(false);

  useEffect(() => {
    if (!GA_ID) return;

    // Check if already consented on mount
    const stored = localStorage.getItem("cookie_consent");
    if (stored === "accepted") {
      setConsented(true);
    }

    // Listen for runtime consent event from CookieBanner
    const handler = () => setConsented(true);
    window.addEventListener("cookie_consent_accepted", handler);
    return () => window.removeEventListener("cookie_consent_accepted", handler);
  }, []);

  if (!consented || !GA_ID) return null;

  return (
    <>
      <Script
        src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}
        strategy="afterInteractive"
      />
      <Script id="ga4-init" strategy="afterInteractive">
        {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', '${GA_ID}', {
            anonymize_ip: true,
            page_path: window.location.pathname,
          });
        `}
      </Script>
    </>
  );
}

/**
 * Helper to fire GA4 custom events from anywhere in the app.
 * Item #19 — custom event tracking.
 */
export function trackEvent(
  eventName: string,
  params?: Record<string, string | number>
) {
  if (typeof window !== "undefined" && typeof (window as any).gtag === "function") {
    (window as any).gtag("event", eventName, params);
  }
}
