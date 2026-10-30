/**
 * Home Page — Overview Dashboard
 * Items: #6 (meta), #10 (alt text on images/icons), #20 (CTA)
 * Mirrors the original app.py main() layout.
 */
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Overview — Phishing Detector Overfitting Study",
  description:
    "Project overview: end-to-end ML pipeline for phishing email detection. 17,522 emails, 4 classifiers, leakage-free evaluation.",
};

// Server Component — fetches reports from the FastAPI backend
async function getReports() {
  try {
    const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const [dataRes, trainRes] = await Promise.all([
      fetch(`${API}/api/reports/data`, { next: { revalidate: 3600 } }),
      fetch(`${API}/api/reports/training`, { next: { revalidate: 3600 } }),
    ]);
    const dataRep = dataRes.ok ? await dataRes.json() : null;
    const trainRep = trainRes.ok ? await trainRes.json() : null;
    return { dataRep, trainRep };
  } catch {
    return { dataRep: null, trainRep: null };
  }
}

export default async function HomePage() {
  const { dataRep, trainRep } = await getReports();

  const cleanRows  = dataRep?.cleaning_statistics?.final_clean_rows ?? 17522;
  const phishPct   = dataRep?.partition_statistics?.train_phishing_pct ?? 37.4;
  const bestModel  = trainRep?.best_model?.key?.replace(/_/g, " ") ?? "Logistic Regression";
  const testAcc    = trainRep?.best_model?.test_metrics?.test_accuracy
    ? (trainRep.best_model.test_metrics.test_accuracy * 100).toFixed(1) + "%"
    : "97.3%";

  return (
    <div className="page-wrapper">
      <div className="container">

        {/* Hero Banner */}
        <section className="hero" aria-labelledby="hero-title">
          <h1 id="hero-title" className="hero-title">
            🛡️ Phishing Message Detector — Overfitting Study
          </h1>
          <p className="hero-subtitle">
            An end-to-end Machine Learning cybersecurity application evaluating email text
            classification, decision boundaries, model complexity, and empirical regularization
            techniques.
          </p>
          <div className="research-callout" role="note">
            <strong>🔬 Central Research Question:</strong>
            <br />
            <em>
              &ldquo;How does model complexity affect phishing message detection, and which techniques
              help reduce overfitting while maintaining high generalization on unseen messages?&rdquo;
            </em>
          </div>
        </section>

        {/* Key Metrics */}
        <section className="section" aria-labelledby="metrics-title">
          <h2 id="metrics-title" className="section-title">📊 Dataset &amp; Performance Telemetry</h2>
          <div className="grid-4">
            <div className="metric-card" aria-label={`Clean corpus: ${cleanRows.toLocaleString()} emails`}>
              <span className="metric-val">{cleanRows.toLocaleString()}</span>
              <span className="metric-lbl">Clean Corpus Emails</span>
            </div>
            <div className="metric-card" aria-label={`Phishing class ratio: ${phishPct}%`}>
              <span className="metric-val" style={{ color: "var(--color-danger)" }}>{phishPct}%</span>
              <span className="metric-lbl">Phishing Class Ratio</span>
            </div>
            <div className="metric-card" aria-label={`Top generalizing model: ${bestModel}`}>
              <span className="metric-val" style={{ color: "var(--color-success)", fontSize: "1.2rem", textTransform: "capitalize" }}>
                {bestModel}
              </span>
              <span className="metric-lbl">Top Generalizer</span>
            </div>
            <div className="metric-card" aria-label={`Test set accuracy: ${testAcc}`}>
              <span className="metric-val">{testAcc}</span>
              <span className="metric-lbl">Test Set Accuracy</span>
            </div>
          </div>
        </section>

        {/* Pipeline Workflow */}
        <section className="section" aria-labelledby="pipeline-title">
          <h2 id="pipeline-title" className="section-title">🔄 Leakage-Free ML Engineering Architecture</h2>
          <p className="mb-lg">
            To guarantee authentic scientific evaluation, the feature extractor (TF-IDF) is strictly fitted
            on the <strong>training split only</strong>. Validation and test partitions remain completely
            unseen during tokenization and IDF weighting.
          </p>
          <div className="grid-6" role="list" aria-label="ML pipeline steps">
            {[
              { step: "1", icon: "📥", title: "Raw Dataset", sub: "HuggingFace Corpus" },
              { step: "2", icon: "🧹", title: "Cleaning & Dedup", sub: "Exact Drop (−1,109)" },
              { step: "3", icon: "🔀", title: "Stratified Split", sub: "70% / 15% / 15%" },
              { step: "4", icon: "🔤", title: "TF-IDF Fit", sub: "Train Set Only" },
              { step: "5", icon: "⚙️", title: "Pipeline Train", sub: "4 Classifiers" },
              { step: "6", icon: "🎯", title: "Prediction UI", sub: "Inference Pipeline" },
            ].map((s) => (
              <div key={s.step} className="flow-step" role="listitem" aria-label={`Step ${s.step}: ${s.title} — ${s.sub}`}>
                <span aria-hidden="true">{s.icon} {s.step}. {s.title}</span>
                <small>{s.sub}</small>
              </div>
            ))}
          </div>
        </section>

        {/* Conceptual Framework */}
        <section className="section" aria-labelledby="concepts-title">
          <h2 id="concepts-title" className="section-title">📚 Conceptual Framework: Overfitting &amp; Generalization</h2>
          <div className="grid-2">
            <div className="concept-card">
              <div className="concept-title">⚠️ Overfitting (High Variance)</div>
              <p>
                Occurs when an overly complex model (e.g. an unconstrained Decision Tree with 50+ depth)
                memorizes noise, exact recipient names, and specific URLs in the training split.
              </p>
              <p className="mt-md">
                <strong>Diagnostic Signature:</strong> Near-perfect training accuracy (99.8%) paired
                with a sharply degraded validation accuracy (85.2%), resulting in a wide{" "}
                <strong>Generalization Gap</strong>.
              </p>
            </div>
            <div className="concept-card">
              <div className="concept-title">🛡️ Regularization &amp; Complexity Control</div>
              <p>
                Mathematical constraints penalizing extreme model parameters to force smoother
                decision boundaries.
              </p>
              <p className="mt-md"><strong>Methods Tested:</strong></p>
              <ul style={{ paddingLeft: "1.2rem", marginTop: 8, color: "var(--color-muted)" }}>
                <li><strong>L2 Regularization (Ridge):</strong> Drives feature weights toward zero.</li>
                <li><strong>L1 Regularization (Lasso):</strong> Enforces weight sparsity.</li>
                <li><strong>Tree Pruning:</strong> Caps depth / leaf size to prevent noise isolation.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Navigation Cards */}
        <section className="section" aria-labelledby="explore-title">
          <h2 id="explore-title" className="section-title">🗺️ Explore the Application</h2>
          <div className="grid-3">
            {[
              { href: "/detector", icon: "🔍", title: "Phishing Detector", desc: "Test real email & SMS messages with live probability scores and heuristic trigger breakdown." },
              { href: "/comparison", icon: "⚖️", title: "Model Comparison", desc: "Side-by-side performance audit across all 4 classifiers with confusion matrices and generalization gaps." },
              { href: "/lab", icon: "🧪", title: "Overfitting Lab", desc: "Adjust Decision Tree depth and Logistic Regression C values to observe generalization curves in real-time." },
              { href: "/errors", icon: "🔬", title: "Error Analysis", desc: "Inspect false positives, false negatives, and token-level log-odds explainability." },
              { href: "/methodology", icon: "📖", title: "Methodology", desc: "Complete scientific documentation, dataset licensing, and experimental conclusions." },
              { href: "/comparison", icon: "📊", title: "Results Summary", desc: "Final test-set verification of the best generalizing model with full classification report." },
            ].map((card) => (
              <Link key={card.href} href={card.href} className="card" style={{ display: "block", textDecoration: "none" }} aria-label={`${card.title}: ${card.desc}`}>
                <div className="concept-title">{card.icon} {card.title}</div>
                <p style={{ fontSize: "0.9rem", marginTop: 8 }}>{card.desc}</p>
              </Link>
            ))}
          </div>
        </section>

        {/* CTA — Item #20 */}
        <section className="cta-section" aria-labelledby="cta-title">
          <h2 id="cta-title">Ready to Test a Message?</h2>
          <p>
            Paste any email or SMS into the live detector and see the TF-IDF pipeline classify it
            in real-time with full heuristic and token-level explanations.
          </p>
          <Link href="/detector" className="btn btn-primary" aria-label="Try the live phishing detector">
            🚀 Try the Live Detector →
          </Link>
        </section>

      </div>
    </div>
  );
}
