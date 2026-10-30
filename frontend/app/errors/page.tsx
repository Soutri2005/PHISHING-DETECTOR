import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Error Analysis & Model Explainability",
  description: "Diagnose false positives, false negatives, and token-level log-odds weights from the best phishing detection model.",
};

export default function ErrorsPage() {
  return (
    <div className="page-wrapper">
      <div className="container">
        <h1>🔬 Error Analysis &amp; Model Explainability</h1>
        <p className="mt-sm mb-lg">
          Diagnose where and why word-based statistical models make mistakes.
          Understanding edge cases is critical for real-world cybersecurity deployments.
        </p>

        <div className="info-box mb-lg">
          💡 <strong>Note:</strong> This page requires a trained model artifact. Run{" "}
          <code>python -m src.train</code> to generate model files, then ensure the FastAPI
          backend is running.
        </div>

        {/* Attack Vectors Section — static content */}
        <section className="section" aria-labelledby="av-title">
          <h2 id="av-title" className="section-title">🛡️ Vulnerabilities of Bag-of-Words / TF-IDF Classifiers</h2>
          <p className="mb-lg">
            While TF-IDF + Logistic Regression achieves impressive benchmark accuracy (~97.8%),
            production cybersecurity systems face distinct adversarial threats:
          </p>
          <div className="grid-2">
            <div className="card">
              <h3 style={{ color: "var(--color-accent)", marginBottom: 12 }}>1. Lexical Evasion &amp; Obfuscation</h3>
              <ul style={{ paddingLeft: "1.2rem", color: "var(--color-muted)", lineHeight: 1.8 }}>
                <li><strong>Homoglyph substitution:</strong> Replacing latin 'o' with cyrillic 'о' to bypass token matching.</li>
                <li><strong>Zero-width characters:</strong> Inserting hidden unicode spaces breaks TF-IDF tokens.</li>
              </ul>
              <h3 style={{ color: "var(--color-accent)", marginBottom: 12, marginTop: 16 }}>2. Context Blindness</h3>
              <p>
                &ldquo;I received a phishing email claiming my account was suspended&rdquo; contains
                high-risk n-grams, causing a <strong>False Positive</strong> on benign threat reports.
              </p>
            </div>
            <div className="card">
              <h3 style={{ color: "var(--color-accent)", marginBottom: 12 }}>3. Image-Based &amp; QR Code Phishing</h3>
              <p>
                Attackers embed the entire malicious lure inside an image or QR code with no
                textual body. A pure NLP model receives empty text and fails completely.
              </p>
              <h3 style={{ color: "var(--color-accent)", marginBottom: 12, marginTop: 16 }}>4. Enterprise IT Disguises</h3>
              <p>
                Real password reset emails from authentic corporate IT use identical urgency
                vocabulary (&ldquo;Expires in 24 hours&rdquo;). Without SPF/DKIM/DMARC header
                verification, content models cannot verify sender provenance.
              </p>
            </div>
          </div>
        </section>

        {/* CTA — Item #20 */}
        <section className="cta-section" aria-labelledby="errors-cta">
          <h2 id="errors-cta">Read the Full Methodology</h2>
          <p>Explore the complete experimental protocol, dataset provenance, and scientific findings.</p>
          <Link href="/methodology" className="btn btn-primary">📖 View Methodology →</Link>
        </section>

      </div>
    </div>
  );
}
