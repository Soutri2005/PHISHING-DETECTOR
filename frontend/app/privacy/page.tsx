import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "Privacy policy for the Phishing Detector Overfitting Study web application. Learn what data we collect and how we handle it.",
};

const LAST_UPDATED = "26 October 2026";

export default function PrivacyPage() {
  return (
    <div className="page-wrapper">
      <div className="container">
        <div className="legal-page">
          <h1>🔒 Privacy Policy</h1>
          <p className="text-muted" style={{ marginBottom: "var(--space-sm)" }}>
            Last updated: {LAST_UPDATED}
          </p>
          <p>
            This Privacy Policy explains how the <strong>Phishing Detector — Overfitting Study</strong>{" "}
            web application (&ldquo;we&rdquo;, &ldquo;our&rdquo;, &ldquo;the app&rdquo;) handles information
            when you use our research tool.
          </p>

          <h2>1. What Data We Collect</h2>

          <h3>1.1 Message Input</h3>
          <p>
            When you submit a message through the Phishing Detector, the text is sent to our FastAPI
            backend for classification. <strong>We do not store, log, or retain any message content
            you submit.</strong> All processing is stateless and in-memory only.
          </p>

          <h3>1.2 Analytics (Optional — Requires Consent)</h3>
          <p>
            If you accept cookies, we use <strong>Google Analytics 4 (GA4)</strong> to collect
            anonymous usage statistics, including:
          </p>
          <ul>
            <li>Pages visited and navigation flow</li>
            <li>General geographic region (country-level only)</li>
            <li>Browser and device type</li>
            <li>Which ML model was selected for inference</li>
          </ul>
          <p>
            IP addresses are anonymized. We do <strong>not</strong> collect your name, email address,
            or any personally identifiable information. GA4 is loaded only after you explicitly click
            &ldquo;Accept All&rdquo; in the cookie banner.
          </p>

          <h3>1.3 Cookies</h3>
          <p>
            We use a single <strong>functional cookie</strong> (<code>cookie_consent</code>) stored
            in <code>localStorage</code> to remember your cookie preference. If you reject analytics,
            no third-party cookies are set.
          </p>

          <h2>2. How We Use Data</h2>
          <ul>
            <li>To classify your submitted message as phishing or legitimate using our ML model.</li>
            <li>To understand aggregate usage patterns and improve the research application.</li>
          </ul>
          <p>We do <strong>not</strong> sell, rent, or share any data with third parties beyond Google Analytics.</p>

          <h2>3. Data Retention</h2>
          <p>
            Message content: <strong>Not retained</strong> — discarded immediately after inference.<br />
            Analytics data: Retained by Google for up to 14 months per default GA4 settings.
          </p>

          <h2>4. Third-Party Services</h2>
          <ul>
            <li>
              <strong>Google Analytics 4</strong> — Used for aggregate anonymous usage analytics.
              See{" "}
              <a href="https://policies.google.com/privacy" target="_blank" rel="noopener noreferrer">
                Google&apos;s Privacy Policy
              </a>.
            </li>
            <li>
              <strong>HuggingFace (Dataset Source)</strong> — The ML model was trained on a
              publicly available dataset. No user data is sent to HuggingFace at runtime.
            </li>
          </ul>

          <h2>5. Your Rights</h2>
          <p>You may at any time:</p>
          <ul>
            <li>Clear your browser&apos;s localStorage to reset your cookie consent preference.</li>
            <li>Use the application with analytics rejected — all core functionality remains available.</li>
          </ul>

          <h2>6. Security</h2>
          <p>
            Communications between your browser and this application are protected by HTTPS
            (TLS encryption). The backend API enforces rate limiting and input length restrictions
            to prevent abuse.
          </p>

          <h2>7. Research Use Disclaimer</h2>
          <p>
            This application is a <strong>research and educational tool</strong>. It is not intended
            for use as a production security system. Do not submit sensitive personal information
            (passwords, credit card numbers, SSNs) through this interface.
          </p>

          <h2>8. Contact</h2>
          <p>
            For privacy questions, please review our{" "}
            <Link href="/terms">Terms &amp; Conditions</Link> or raise an issue on the project
            repository.
          </p>

          <div className="info-box mt-xl">
            By using this application, you acknowledge that you have read and understood this
            Privacy Policy.
          </div>
        </div>
      </div>
    </div>
  );
}
