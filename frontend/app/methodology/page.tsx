import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Methodology & Empirical Findings",
  description: "Detailed research methodology, experimental protocols, data leakage prevention audits, and empirical conclusions from the Phishing Message Detector Overfitting Study.",
};

export default function MethodologyPage() {
  return (
    <div className="page-wrapper">
      <div className="container">
        <h1>📖 Methodology &amp; Empirical Findings</h1>
        <p className="mt-sm mb-lg">
          Detailed research methodology, experimental protocols, data leakage prevention audits,
          and empirical conclusions from the Phishing Message Detector Overfitting Study.
        </p>

        {/* Tabs — static rendered for SEO */}
        <section className="section" aria-labelledby="dataset-title">
          <h2 id="dataset-title" className="section-title">📁 Dataset &amp; Data Hygiene</h2>
          <p className="mb-md">
            The study employs the publicly available{" "}
            <strong>Phishing Email Dataset</strong> from Hugging Face:{" "}
            <a
              href="https://huggingface.co/datasets/zefang-liu/phishing-email-dataset"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Phishing Email Dataset on HuggingFace (opens in new tab)"
            >
              zefang-liu/phishing-email-dataset
            </a>.
          </p>
          <div className="card mb-lg">
            <h3 style={{ color: "var(--color-accent)", marginBottom: 12 }}>🧹 Data Cleaning Pipeline</h3>
            <ol style={{ paddingLeft: "1.2rem", color: "var(--color-muted)", lineHeight: 2 }}>
              <li><strong>Header &amp; Column Normalization:</strong> Standardized raw columns to <code>text</code> and <code>label</code>.</li>
              <li><strong>Label Canonicalization:</strong> Mapped string classes (<code>&ldquo;Phishing Email&rdquo; → 1</code>, <code>&ldquo;Safe Email&rdquo; → 0</code>).</li>
              <li><strong>Missing Value Removal:</strong> Dropped null/empty strings and non-printable ASCII control characters.</li>
              <li><strong>Pre-Split Exact Deduplication:</strong> Removed 1,109 exact duplicate emails <em>before</em> splitting to prevent overlap.</li>
            </ol>
          </div>
        </section>

        <section className="section" aria-labelledby="leakage-title">
          <h2 id="leakage-title" className="section-title">🔒 Leakage Prevention Protocol</h2>
          <p className="mb-md">
            Data leakage occurs when information from validation or test sets contaminates training,
            yielding artificially inflated metrics that fail in production.
          </p>
          <div className="card">
            <h3 style={{ color: "var(--color-accent)", marginBottom: 12 }}>🛡️ Strict Leakage Prevention Controls</h3>
            <ol style={{ paddingLeft: "1.2rem", color: "var(--color-muted)", lineHeight: 2 }}>
              <li><strong>Stratified Partitioning:</strong> 70% Train / 15% Validation / 15% Test. Class distribution locked at <strong>37.35% Phishing</strong> across all three splits.</li>
              <li><strong>Feature Fit Isolation:</strong> <code>TfidfVectorizer</code> fitted <strong>only on X_train</strong>. IDF weights never see validation or test tokens.</li>
              <li><strong>Zero Cross-Partition Overlap:</strong> Automated set-intersection audit verified exactly <strong>0 messages</strong> overlap between partitions.</li>
              <li><strong>Single-Touch Test Evaluation:</strong> Test split touched exactly once after model selection. Hyperparameters never tuned against test scores.</li>
            </ol>
          </div>
          <div className="success-box mt-md">
            ✅ <strong>Leakage Audit Status:</strong> PASSED — Zero overlapping texts across all 3 partitions.
          </div>
        </section>

        <section className="section" aria-labelledby="findings-title">
          <h2 id="findings-title" className="section-title">🧠 Scientific Findings on Overfitting in NLP</h2>
          <div className="finding-highlight">
            <h3 style={{ marginBottom: 8 }}>1. The Decision Tree Complexity Paradox</h3>
            <p>
              An unconstrained Decision Tree achieved <strong>99.99% training accuracy</strong> but
              fell to <strong>93.46% validation accuracy</strong>, exhibiting a massive{" "}
              <strong>6.53% Generalization Gap</strong>. Because text data has 5,000 sparse features,
              greedy orthogonal tree splits easily isolate idiosyncratic words into single-email leaves.
            </p>
          </div>
          <div className="finding-highlight">
            <h3 style={{ marginBottom: 8 }}>2. Regularization Superiority in Linear Models</h3>
            <p>
              Logistic Regression with L₂ regularization achieved <strong>97.79% validation accuracy</strong>{" "}
              with only a <strong>0.82% gap</strong> (Train: 98.61%). By penalizing extreme coefficients,
              the model learns smooth, generalized decision boundaries across all vocabulary tokens.
            </p>
          </div>
          <div className="finding-highlight">
            <h3 style={{ marginBottom: 8 }}>3. Ensembling Mitigates Variance</h3>
            <p>
              Random Forest reduced the Decision Tree generalization gap from{" "}
              <strong>6.53% down to 3.11%</strong> while boosting validation accuracy to{" "}
              <strong>96.88%</strong>. Bagging and random feature subspace selection smooth out
              high-variance decision surfaces.
            </p>
          </div>
        </section>

        <section className="section" aria-labelledby="limits-title">
          <h2 id="limits-title" className="section-title">🎯 Limitations &amp; Future Work</h2>
          <div className="grid-2">
            <div className="card">
              <h3 style={{ color: "var(--color-accent)", marginBottom: 12 }}>Technical Limitations</h3>
              <ul style={{ paddingLeft: "1.2rem", color: "var(--color-muted)", lineHeight: 1.9 }}>
                <li><strong>Header Ignorance:</strong> Body-only classification cannot inspect SMTP hops, DKIM signatures, or DMARC alignment.</li>
                <li><strong>Adversarial Perturbations:</strong> Attackers can bypass word models using zero-width unicode spaces or image-rendered text.</li>
                <li><strong>Temporal Drift:</strong> New phishing lures and AI-generated spear-phishing shift the underlying distribution over time.</li>
              </ul>
            </div>
            <div className="card">
              <h3 style={{ color: "var(--color-accent)", marginBottom: 12 }}>Future Enhancements</h3>
              <ul style={{ paddingLeft: "1.2rem", color: "var(--color-muted)", lineHeight: 1.9 }}>
                <li><strong>Subword Tokenization:</strong> Replace TF-IDF with BPE or lightweight DistilBERT embeddings.</li>
                <li><strong>Multi-Modal OCR:</strong> Extract text from embedded images and scan QR codes automatically.</li>
                <li><strong>Header Telemetry:</strong> Integrate SPF, DKIM, and Domain Age WHOIS lookup into the feature vector.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* CTA — Item #20 */}
        <section className="cta-section" aria-labelledby="methodology-cta">
          <h2 id="methodology-cta">Try the Live Detector</h2>
          <p>Put the methodology into practice — classify any message using the fitted TF-IDF pipeline.</p>
          <Link href="/detector" className="btn btn-primary">🚀 Try the Live Detector →</Link>
        </section>

      </div>
    </div>
  );
}
