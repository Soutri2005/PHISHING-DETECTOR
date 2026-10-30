/**
 * Custom 404 page — Item #15
 * Branded 404 with cyber theme and navigation guidance.
 */
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "404 — Page Not Found",
  description: "The page you are looking for does not exist. Return to the Phishing Detector dashboard.",
};

export default function NotFound() {
  return (
    <div className="page-wrapper">
      <div className="page-404" role="main" aria-labelledby="not-found-title">
        <div className="error-code" aria-hidden="true">404</div>

        <h1 id="not-found-title" style={{ marginTop: 16, marginBottom: 8 }}>
          Page Not Found
        </h1>
        <p style={{ maxWidth: 480, marginBottom: 32, color: "var(--color-muted)" }}>
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
          Let&apos;s get you back to the dashboard.
        </p>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
          <Link href="/" className="btn btn-primary" aria-label="Return to the dashboard">
            🏠 Return to Dashboard
          </Link>
          <Link href="/detector" className="btn btn-secondary" aria-label="Try the phishing detector">
            🔍 Try the Detector
          </Link>
        </div>

        <div style={{ marginTop: 48, color: "var(--color-muted)", fontSize: "0.88rem" }}>
          Looking for something specific?{" "}
          <Link href="/methodology">Methodology</Link> ·{" "}
          <Link href="/comparison">Model Comparison</Link> ·{" "}
          <Link href="/lab">Overfitting Lab</Link>
        </div>
      </div>
    </div>
  );
}
