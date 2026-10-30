"use client";

/**
 * Page 2: Model Comparison Dashboard
 * Server-side data fetched from FastAPI /api/reports/training.
 * Charts via Recharts. Items: #6 (meta), #10 (chart aria labels), #20 (CTA)
 */
import type { Metadata } from "next";
import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell
} from "recharts";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ModelData {
  train_accuracy: number;
  val_accuracy: number;
  generalization_gap: number;
  val_precision: number;
  val_recall: number;
  val_f1: number;
  training_time_seconds: number;
  val_confusion_matrix: { tp: number; tn: number; fp: number; fn: number };
}

interface TrainingReport {
  baseline_models: Record<string, ModelData>;
  best_model: {
    key: string;
    test_metrics: {
      test_accuracy: number;
      test_recall: number;
      test_f1: number;
    };
  };
}

function toLabel(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const COLORS = ["#38bdf8", "#10b981", "#f59e0b", "#a78bfa"];
const DANGER_COLORS = ["#ef4444", "#f97316", "#eab308", "#22d3ee"];

export default function ComparisonPage() {
  const [report, setReport] = useState<TrainingReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/reports/training`)
      .then((r) => {
        if (!r.ok) throw new Error(`API returned ${r.status}`);
        return r.json();
      })
      .then(setReport)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page-wrapper" style={{ textAlign: "center", paddingTop: 80 }}>
        <span className="spinner" aria-label="Loading model comparison data" style={{ width: 40, height: 40 }}></span>
        <p className="mt-md">Loading model comparison data…</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="page-wrapper">
        <div className="container">
          <div className="disclaimer" role="alert">
            <strong>⚠ Data unavailable:</strong> {error || "Training report not found."}<br />
            <small>Run <code>python -m src.train</code> then restart the FastAPI backend.</small>
          </div>
        </div>
      </div>
    );
  }

  const modelEntries = Object.entries(report.baseline_models);

  const accChartData = modelEntries.map(([key, data]) => ({
    name: toLabel(key),
    "Train Accuracy": parseFloat((data.train_accuracy * 100).toFixed(2)),
    "Val Accuracy": parseFloat((data.val_accuracy * 100).toFixed(2)),
  }));

  const gapChartData = modelEntries.map(([key, data]) => ({
    name: toLabel(key),
    "Gen. Gap (%)": parseFloat((data.generalization_gap * 100).toFixed(2)),
  }));

  const bestName = toLabel(report.best_model.key);
  const testMet  = report.best_model.test_metrics;

  return (
    <div className="page-wrapper">
      <div className="container">
        <h1>⚖️ Baseline Model Comparison</h1>
        <p className="mt-sm mb-lg">
          Direct comparison of four real machine learning classifiers trained on identical
          training partitions and evaluated on the held-out validation and test sets.
        </p>

        {/* Performance Table */}
        <section className="section" aria-labelledby="table-title">
          <h2 id="table-title" className="section-title">📊 Performance Benchmark Table</h2>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table" aria-label="Model performance benchmark comparison">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Train Acc.</th>
                  <th scope="col">Val Acc.</th>
                  <th scope="col">Gen. Gap</th>
                  <th scope="col">Val Precision</th>
                  <th scope="col">Val Recall</th>
                  <th scope="col">Val F1</th>
                  <th scope="col">Train Time</th>
                </tr>
              </thead>
              <tbody>
                {modelEntries.map(([key, data]) => (
                  <tr key={key}>
                    <td><strong>{toLabel(key)}</strong></td>
                    <td>{(data.train_accuracy * 100).toFixed(2)}%</td>
                    <td>{(data.val_accuracy * 100).toFixed(2)}%</td>
                    <td>
                      <span className={`badge ${data.generalization_gap > 0.05 ? "badge-danger" : data.generalization_gap > 0.02 ? "badge-warning" : "badge-success"}`}>
                        {(data.generalization_gap * 100).toFixed(2)}%
                      </span>
                    </td>
                    <td>{(data.val_precision * 100).toFixed(2)}%</td>
                    <td>{(data.val_recall * 100).toFixed(2)}%</td>
                    <td>{(data.val_f1 * 100).toFixed(2)}%</td>
                    <td>{data.training_time_seconds.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Charts */}
        <section className="section" aria-labelledby="charts-title">
          <h2 id="charts-title" className="section-title">📈 Diagnostic Charts</h2>
          <div className="grid-2">
            {/* Accuracy Chart */}
            <div className="card">
              <h3 style={{ marginBottom: 16 }}>Training vs Validation Accuracy</h3>
              <div aria-label="Grouped bar chart comparing training and validation accuracy for each model">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={accChartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis domain={[80, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9" }}
                      formatter={(v) => [`${v ?? 0}%`]}
                    />
                    <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                    <Bar dataKey="Train Accuracy" fill="#06b6d4" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="Val Accuracy" fill="#10b981" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Gap Chart */}
            <div className="card">
              <h3 style={{ marginBottom: 16 }}>Generalization Gap (Higher = More Overfitted)</h3>
              <div aria-label="Bar chart showing generalization gap per model — higher values indicate more overfitting">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={gapChartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9" }}
                      formatter={(v) => [`${v ?? 0}%`, "Gen. Gap"]}
                    />
                    <Bar dataKey="Gen. Gap (%)" radius={[3, 3, 0, 0]}>
                      {gapChartData.map((_, i) => (
                        <Cell key={i} fill={DANGER_COLORS[i % DANGER_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </section>

        {/* Confusion Matrices */}
        <section className="section" aria-labelledby="cm-title">
          <h2 id="cm-title" className="section-title">🎯 Confusion Matrices &amp; Error Trade-offs</h2>
          <div className="info-box mb-md">
            <strong>In cybersecurity, not all errors are equal:</strong>
            <br />• <strong>False Positive (FP):</strong> Legitimate email flagged as phishing — causes annoyance and communication failure.
            <br />• <strong>False Negative (FN):</strong> Phishing email missed — causes credential theft, ransomware, and breach.
          </div>
          <div className="grid-4">
            {modelEntries.map(([key, data]) => {
              const cm = data.val_confusion_matrix;
              const name = toLabel(key);
              return (
                <div key={key} className="card">
                  <h3 style={{ fontSize: "0.95rem", marginBottom: 12 }}>{name}</h3>
                  <table className="cm-table" aria-label={`Confusion matrix for ${name}`}>
                    <thead>
                      <tr>
                        <th style={{ color: "var(--color-muted)", fontSize: "0.72rem", padding: 6 }}>Actual \ Pred</th>
                        <th style={{ color: "var(--color-muted)", fontSize: "0.72rem", padding: 6 }}>Legitimate</th>
                        <th style={{ color: "var(--color-muted)", fontSize: "0.72rem", padding: 6 }}>Phishing</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ color: "var(--color-muted)", fontSize: "0.72rem", padding: 6 }}>Legit</td>
                        <td className="cm-cell cm-tn">TN: {cm.tn}</td>
                        <td className="cm-cell cm-fp">FP: {cm.fp}</td>
                      </tr>
                      <tr>
                        <td style={{ color: "var(--color-muted)", fontSize: "0.72rem", padding: 6 }}>Phish</td>
                        <td className="cm-cell cm-fn">FN: {cm.fn}</td>
                        <td className="cm-cell cm-tp">TP: {cm.tp}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              );
            })}
          </div>
        </section>

        {/* Best Model Final Test */}
        <section className="section" aria-labelledby="test-title">
          <h2 id="test-title" className="section-title">🔒 Untouched Test Set Verification (Best Model)</h2>
          <div className="grid-4">
            <div className="metric-card"><span className="metric-val" style={{ fontSize: "1.1rem", textTransform: "capitalize" }}>{bestName}</span><span className="metric-lbl">Final Model</span></div>
            <div className="metric-card"><span className="metric-val">{(testMet.test_accuracy * 100).toFixed(2)}%</span><span className="metric-lbl">Test Accuracy</span></div>
            <div className="metric-card"><span className="metric-val" style={{ color: "var(--color-success)" }}>{(testMet.test_recall * 100).toFixed(2)}%</span><span className="metric-lbl">Test Recall (Catch Rate)</span></div>
            <div className="metric-card"><span className="metric-val">{(testMet.test_f1 * 100).toFixed(2)}%</span><span className="metric-lbl">Test F1-Score</span></div>
          </div>
          <div className="success-box mt-md">
            ✅ <strong>Methodology Check:</strong> The test set was touched exactly once after model
            selection to verify generalization. Hyperparameters were never tuned against these test scores.
          </div>
        </section>

        {/* CTA — Item #20 */}
        <section className="cta-section" aria-labelledby="comparison-cta">
          <h2 id="comparison-cta">Explore the Overfitting Lab</h2>
          <p>Interactively tune Decision Tree depth and Logistic Regression C to watch generalization curves shift in real-time.</p>
          <Link href="/lab" className="btn btn-primary">🧪 Open Overfitting Lab →</Link>
        </section>

      </div>
    </div>
  );
}
