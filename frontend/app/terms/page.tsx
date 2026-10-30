import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Terms & Conditions",
  description: "Terms and conditions for using the Phishing Detector Overfitting Study application. Research use only. No liability for classification accuracy.",
};

const LAST_UPDATED = "26 October 2026";

export default function TermsPage() {
  return (
    <div className="page-wrapper">
      <div className="container">
        <div className="legal-page">
          <h1>📄 Terms &amp; Conditions</h1>
          <p className="text-muted" style={{ marginBottom: "var(--space-sm)" }}>
            Last updated: {LAST_UPDATED}
          </p>
          <p>
            Please read these Terms &amp; Conditions (&ldquo;Terms&rdquo;) carefully before using
            the <strong>Phishing Detector — Overfitting Study</strong> application.
          </p>

          <h2>1. Acceptance of Terms</h2>
          <p>
            By accessing or using this application, you agree to be bound by these Terms. If you
            do not agree, please discontinue use immediately.
          </p>

          <h2>2. Research &amp; Educational Use Only</h2>
          <p>
            This application is provided strictly for <strong>research, educational, and
            demonstration purposes</strong>. It is a study of machine learning overfitting
            phenomena using a publicly available phishing email dataset.
          </p>
          <p>
            It is <strong>not intended to be relied upon as a production cybersecurity system</strong>.
            Do not use this tool as your sole method of determining whether an email is safe or
            malicious in a real-world context.
          </p>

          <h2>3. No Accuracy Warranty</h2>
          <p>
            The machine learning models used in this application are statistical estimators trained
            on a specific dataset. They will produce errors, including:
          </p>
          <ul>
            <li><strong>False Positives:</strong> Legitimate emails incorrectly classified as phishing.</li>
            <li><strong>False Negatives:</strong> Malicious emails incorrectly classified as legitimate.</li>
          </ul>
          <p>
            We provide <strong>no warranty, express or implied</strong>, regarding the accuracy,
            reliability, or completeness of any classification result.
          </p>

          <h2>4. No Liability</h2>
          <p>
            To the fullest extent permitted by applicable law, the creators and contributors of this
            application shall not be liable for any damages arising out of or in connection with
            your use of, or inability to use, this application — including but not limited to direct,
            indirect, incidental, punitive, or consequential damages.
          </p>

          <h2>5. Acceptable Use Policy</h2>
          <p>You agree <strong>not</strong> to:</p>
          <ul>
            <li>Submit personally identifiable information (passwords, SSNs, credit card numbers) to the message analysis form.</li>
            <li>Attempt to reverse-engineer, extract, or steal the trained model artifacts.</li>
            <li>Use automated scripts to abuse or overload the prediction API beyond reasonable research use.</li>
            <li>Misrepresent the capabilities of this tool to others.</li>
          </ul>

          <h2>6. Intellectual Property</h2>
          <p>
            The application code is provided for educational purposes. The training dataset is
            subject to its own license:{" "}
            <a
              href="https://huggingface.co/datasets/zefang-liu/phishing-email-dataset"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Phishing Email Dataset license on HuggingFace (opens in new tab)"
            >
              zefang-liu/phishing-email-dataset on HuggingFace
            </a>. Please review dataset license terms before any commercial use.
          </p>

          <h2>7. Privacy</h2>
          <p>
            Your use of this application is also governed by our{" "}
            <Link href="/privacy">Privacy Policy</Link>, which is incorporated into these Terms
            by reference.
          </p>

          <h2>8. Changes to Terms</h2>
          <p>
            We reserve the right to modify these Terms at any time. The &ldquo;Last updated&rdquo;
            date at the top of this page will be revised accordingly. Continued use after changes
            constitutes acceptance of the new Terms.
          </p>

          <h2>9. Governing Law</h2>
          <p>
            These Terms shall be governed by and construed in accordance with applicable laws,
            without regard to conflict of law principles.
          </p>

          <div className="disclaimer mt-xl">
            🛡️ <strong>Reminder:</strong> This is a research tool. Never rely solely on automated
            ML classification for real-world security decisions. When in doubt, consult a qualified
            cybersecurity professional.
          </div>
        </div>
      </div>
    </div>
  );
}
