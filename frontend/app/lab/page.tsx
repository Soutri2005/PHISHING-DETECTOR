"use client";

/**
 * Page 3: Overfitting Lab
 * Fetches pre-computed experiment CSVs via the FastAPI backend.
 * Interactive charts via Recharts. Item #20 (CTA).
 */
import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell,
} from "recharts";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ExpRow = Record<string, number | string>;

function useExperiment(type: string) {
  const [data, setData] = useState<ExpRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch(`${API_URL}/api/experiments/${type}`)
      .then((r) => r.ok ? r.json() : Promise.reject(r.statusText))
      .then((j) => setData(j.data))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [type]);
  return { data, loading, error };
}

const chartStyle = {
  contentStyle: { background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9" },
  tickStyle: { fill: "#94a3b8", fontSize: 11 },
};

export default function LabPage() {
  const [activeTab, setActiveTab] = useState(0);
  const dt = useExperiment("decision_tree");
  const lr = useExperiment("logistic_regression");
  const lc = useExperiment("learning_curve");

  const TABS = [
    { label: "🌲 Decision Tree Complexity", id: "tab-dt" },
    { label: "⚖️ LR Regularization", id: "tab-lr" },
    { label: "📈 Learning Curves", id: "tab-lc" },
  ];

  return (
    <div className="page-wrapper">
      <div className="container">
        <h1>🧪 Interactive Overfitting Lab</h1>
        <p className="mt-sm mb-lg">
          Explore how <strong>model capacity</strong>, <strong>tree depth</strong>, and{" "}
          <strong>regularization penalties</strong> shape the boundary between healthy
          generalization and severe memorization of noise.
        </p>

        {/* Tabs */}
        <div className="tabs" role="tablist" aria-label="Overfitting lab experiments">
          {TABS.map((tab, i) => (
            <button
              key={tab.id}
              id={tab.id}
              role="tab"
              aria-selected={activeTab === i}
              aria-controls={`panel-${i}`}
              className={`tab-btn${activeTab === i ? " active" : ""}`}
              onClick={() => setActiveTab(i)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab 0: Decision Tree */}
        <div id="panel-0" role="tabpanel" aria-labelledby="tab-dt" hidden={activeTab !== 0}>
          <h2 className="section-title">🌲 Experiment A: Tree Depth &amp; Capacity vs Generalization</h2>
          <p className="mb-lg">
            Decision Trees split features greedily. When given <strong>unlimited depth</strong>,
            the tree splits until nearly every training email resides in its own pure leaf node.
            In text classification with 5,000 TF-IDF features, this causes the tree to memorize
            noise words, driving <strong>Training Accuracy to ~100%</strong> while{" "}
            <strong>Validation Accuracy degrades</strong>, yielding a severe <strong>Generalization Gap</strong>.
          </p>

          {dt.loading && <div style={{ textAlign: "center" }}><span className="spinner" aria-label="Loading decision tree data"></span></div>}
          {dt.error && <div className="disclaimer" role="alert">⚠ Could not load experiment data: {dt.error}</div>}

          {!dt.loading && !dt.error && dt.data.length > 0 && (
            <div className="grid-2">
              <div className="card">
                <h3 style={{ marginBottom: 16 }}>Accuracy Divergence: Overfitting Signature</h3>
                <div aria-label="Line chart showing training and validation accuracy diverging as Decision Tree depth increases">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={dt.data.map((r) => ({ depth: r.max_depth_param, Train: +(+r.train_accuracy * 100).toFixed(2), Val: +(+r.val_accuracy * 100).toFixed(2) }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="depth" tick={chartStyle.tickStyle} label={{ value: "Max Depth", position: "insideBottom", fill: "#94a3b8", offset: -5 }} />
                      <YAxis domain={[75, 101]} tick={chartStyle.tickStyle} tickFormatter={(v) => `${v}%`} />
                      <Tooltip contentStyle={chartStyle.contentStyle} formatter={(v) => [`${v ?? 0}%`]} />
                      <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                      <Line type="monotone" dataKey="Train" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} name="Train Accuracy" />
                      <Line type="monotone" dataKey="Val" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Val Accuracy" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="card">
                <h3 style={{ marginBottom: 16 }}>Generalization Gap vs Max Depth</h3>
                <div aria-label="Bar chart showing increasing generalization gap as tree depth increases">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={dt.data.map((r) => ({ depth: r.max_depth_param, gap: +(+r.generalization_gap * 100).toFixed(2) }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="depth" tick={chartStyle.tickStyle} />
                      <YAxis tick={chartStyle.tickStyle} tickFormatter={(v) => `${v}%`} />
                      <Tooltip contentStyle={chartStyle.contentStyle} formatter={(v) => [`${v ?? 0}%`, "Gen. Gap"]} />
                      <Bar dataKey="gap" name="Generalization Gap" radius={[3, 3, 0, 0]}>
                        {dt.data.map((_, i) => <Cell key={i} fill={`hsl(${0 + i * 8}, 80%, 55%)`} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Tab 1: Logistic Regression */}
        <div id="panel-1" role="tabpanel" aria-labelledby="tab-lr" hidden={activeTab !== 1}>
          <h2 className="section-title">⚖️ Experiment B: Regularization Strength C &amp; Sparsity</h2>
          <p className="mb-lg">
            In Logistic Regression, <strong>C</strong> controls the inverse regularization strength.
            Small C → strong penalty → simpler model. Large C → weak penalty → risks overfitting.
            L1 (Lasso) enforces sparsity; L2 (Ridge) shrinks continuously.
          </p>

          {lr.loading && <div style={{ textAlign: "center" }}><span className="spinner" aria-label="Loading LR experiment data"></span></div>}
          {lr.error && <div className="disclaimer" role="alert">⚠ Could not load experiment data: {lr.error}</div>}

          {!lr.loading && !lr.error && lr.data.length > 0 && (() => {
            const l2data = lr.data.filter((r) => r.penalty === "l2");
            const sparseData = lr.data.map((r) => ({ C: String(r.C), penalty: String(r.penalty), weights: +r.non_zero_weights }));
            return (
              <div className="grid-2">
                <div className="card">
                  <h3 style={{ marginBottom: 16 }}>C vs Accuracy (L2 Penalty)</h3>
                  <div aria-label="Line chart showing effect of regularization strength C on training and validation accuracy">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={l2data.map((r) => ({ C: String(r.C), Train: +(+r.train_accuracy * 100).toFixed(2), Val: +(+r.val_accuracy * 100).toFixed(2) }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="C" tick={chartStyle.tickStyle} />
                        <YAxis domain={[80, 100]} tick={chartStyle.tickStyle} tickFormatter={(v) => `${v}%`} />
                        <Tooltip contentStyle={chartStyle.contentStyle} formatter={(v) => [`${v ?? 0}%`]} />
                        <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                        <Line type="monotone" dataKey="Train" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} name="Train Accuracy" />
                        <Line type="monotone" dataKey="Val" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Val Accuracy" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="card">
                  <h3 style={{ marginBottom: 16 }}>Feature Sparsity: L1 vs L2</h3>
                  <div aria-label="Grouped bar chart comparing active non-zero feature counts between L1 and L2 regularization">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={sparseData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="C" tick={chartStyle.tickStyle} />
                        <YAxis tick={chartStyle.tickStyle} />
                        <Tooltip contentStyle={chartStyle.contentStyle} />
                        <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                        <Bar dataKey="weights" name="Non-Zero Features" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>

        {/* Tab 2: Learning Curves */}
        <div id="panel-2" role="tabpanel" aria-labelledby="tab-lc" hidden={activeTab !== 2}>
          <h2 className="section-title">📈 Experiment C: Learning Curves</h2>
          <p className="mb-lg">
            At small sample sizes, training accuracy is artificially high while validation is low.
            As sample size scales, training and validation scores converge toward the true asymptotic capacity.
          </p>

          {lc.loading && <div style={{ textAlign: "center" }}><span className="spinner" aria-label="Loading learning curve data"></span></div>}
          {lc.error && <div className="disclaimer" role="alert">⚠ Could not load experiment data: {lc.error}</div>}

          {!lc.loading && !lc.error && lc.data.length > 0 && (
            <div className="card">
              <h3 style={{ marginBottom: 16 }}>Empirical Learning Curve: Logistic Regression (C=1.0)</h3>
              <div aria-label="Line chart showing training and validation accuracy converging as training sample size increases">
                <ResponsiveContainer width="100%" height={380}>
                  <LineChart data={lc.data.map((r) => ({ Samples: +r.train_samples, Train: +(+r.train_accuracy * 100).toFixed(2), Val: +(+r.val_accuracy * 100).toFixed(2) }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="Samples" tick={chartStyle.tickStyle} label={{ value: "Training Samples", position: "insideBottom", fill: "#94a3b8", offset: -5 }} />
                    <YAxis domain={[80, 101]} tick={chartStyle.tickStyle} tickFormatter={(v) => `${v}%`} />
                    <Tooltip contentStyle={chartStyle.contentStyle} formatter={(v) => [`${v ?? 0}%`]} />
                    <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 12 }} />
                    <Line type="monotone" dataKey="Train" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} name="Training Accuracy" />
                    <Line type="monotone" dataKey="Val" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Validation Accuracy" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* CTA — Item #20 */}
        <section className="cta-section" aria-labelledby="lab-cta">
          <h2 id="lab-cta">Dig Into Error Cases</h2>
          <p>See which emails the best model gets wrong and why — with full token-level explainability.</p>
          <Link href="/errors" className="btn btn-primary">🔬 View Error Analysis →</Link>
        </section>

      </div>
    </div>
  );
}
