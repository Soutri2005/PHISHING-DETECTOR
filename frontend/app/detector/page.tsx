"use client";

/**
 * Page 1: Phishing Detector — Live Inference
 * Items: #17 (form validation), #18 (honeypot), #19 (analytics events), #20 (CTA)
 */
import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import Link from "next/link";
import { trackEvent } from "@/components/AnalyticsScript";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SAMPLE_PRESETS: Record<string, string> = {
  "🏦 High-Risk Bank Phishing":
    "URGENT SECURITY ALERT: We have detected unauthorized attempts to access your Chase Bank account from an unrecognized IP address. " +
    "Your access will be permanently suspended within 24 hours unless you verify your identity immediately. " +
    "Click here to confirm your credentials, password, and SSN: http://192.168.1.105/auth/chase-verify",
  "💻 Urgent IT Helpdesk Spoof":
    "Attention Employee: Your Microsoft Office 365 password expires today. " +
    "Failure to update your password immediately will terminate your corporate email access. " +
    "Please update your login credentials here: http://secure-corporate-auth-update.net/login",
  "🎁 Gift Card Scam":
    "Congratulations! You have been selected as a Walmart $1,000 gift card winner. " +
    "Claim your prize immediately before the offer expires in 12 hours. " +
    "Enter your credit card number and billing address to verify delivery: http://free-gift-card-portal.com",
  "💼 Benign Work Project Email":
    "Hi Alex,\n\nI reviewed the Q3 machine learning project roadmap. The proposed cross-validation and feature extraction steps look solid. " +
    "Let's schedule a 30-minute sync on Thursday at 2:00 PM to finalize the evaluation metrics before our sprint planning meeting.\n\nBest regards,\nDavid",
  "📦 Benign Delivery Confirmation":
    "Thank you for your order! Your package has shipped via FedEx and is scheduled for delivery on Monday. " +
    "You can track your package progress anytime in your account dashboard. Thank you for shopping with us!",
};

const MODEL_OPTIONS = [
  { value: "best", label: "Best Generalizer (Recommended)" },
  { value: "logistic_regression", label: "Logistic Regression" },
  { value: "naive_bayes", label: "Naive Bayes" },
  { value: "decision_tree", label: "Decision Tree" },
  { value: "random_forest", label: "Random Forest" },
];

const MAX_LENGTH = 50000;
const MIN_LENGTH = 10;

interface FormValues {
  message: string;
  model_name: string;
  hp_field: string; // honeypot
}

interface PredictResult {
  prediction: number;
  label: string;
  phishing_probability: number;
  confidence: number;
  model_used: string;
  heuristics: {
    url_count: number;
    has_ip_url: boolean;
    urls_detected: string[];
    urgency_triggers: string[];
    credential_triggers: string[];
    heuristic_risk_score: number;
  };
  influential_tokens: Array<{
    token: string;
    direction: string;
    tfidf: number;
    model_weight: number;
    impact: number;
  }>;
}

export default function DetectorPage() {
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors },
  } = useForm<FormValues>({
    defaultValues: {
      message: SAMPLE_PRESETS["🏦 High-Risk Bank Phishing"],
      model_name: "best",
      hp_field: "",
    },
  });

  const messageVal = watch("message") ?? "";
  const charCount  = messageVal.length;

  const onSubmit = async (data: FormValues) => {
    setLoading(true);
    setApiError(null);
    setResult(null);

    // Item #18: Honeypot check (also validated server-side)
    if (data.hp_field) {
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: data.message,
          model_name: data.model_name,
          hp_field: data.hp_field,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const prediction: PredictResult = await res.json();
      setResult(prediction);

      // Item #19 — GA4 custom event
      trackEvent("detect_phishing_submit", {
        model: data.model_name,
        label: prediction.label,
        confidence: Math.round(prediction.confidence * 100),
      });
    } catch (err: any) {
      setApiError(err.message || "Failed to connect to the prediction API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-wrapper">
      <div className="container">
        <h1>🔍 Live Phishing Message Detector</h1>
        <p className="mt-sm mb-lg">
          Enter any raw email body, text message, or notice. The message is evaluated through the
          fitted TF-IDF vectorizer and classification pipeline with zero refitting.
        </p>

        <div className="grid-2" style={{ gridTemplateColumns: "300px 1fr", alignItems: "start", gap: "var(--space-lg)" }}>

          {/* Sidebar */}
          <aside aria-label="Quick load presets and model selection">
            <div className="card" style={{ position: "sticky", top: "calc(var(--nav-height) + 16px)" }}>
              <h2 style={{ fontSize: "1rem", marginBottom: "var(--space-md)" }}>⚙️ Settings</h2>

              {/* Model selector */}
              <div className="form-group">
                <label className="form-label" htmlFor="model-select">Model Pipeline:</label>
                <select
                  id="model-select"
                  className="form-select"
                  {...register("model_name")}
                  aria-describedby="model-hint"
                  onChange={(e) => {
                    trackEvent("model_selected", { model: e.target.value });
                    setValue("model_name", e.target.value);
                  }}
                >
                  {MODEL_OPTIONS.map((m) => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
                <p id="model-hint" className="form-hint">
                  Select which trained classifier pipeline to use.
                </p>
              </div>

              <hr style={{ borderColor: "var(--color-border)", margin: "var(--space-md) 0" }} />

              <h3 style={{ fontSize: "0.9rem", color: "var(--color-muted)", marginBottom: "var(--space-sm)" }}>
                📋 Quick-Load Test Messages
              </h3>
              {Object.entries(SAMPLE_PRESETS).map(([name, text]) => (
                <button
                  key={name}
                  type="button"
                  className="btn btn-secondary"
                  style={{ width: "100%", marginBottom: 6, fontSize: "0.82rem", padding: "8px 12px", justifyContent: "flex-start" }}
                  onClick={() => {
                    setValue("message", text);
                    trackEvent("sample_preset_loaded", { preset: name });
                  }}
                  aria-label={`Load sample: ${name}`}
                >
                  {name}
                </button>
              ))}
            </div>
          </aside>

          {/* Main Form */}
          <div>
            <form onSubmit={handleSubmit(onSubmit)} noValidate aria-label="Phishing message detection form">

              {/* Item #18: Honeypot — hidden from real users, visible to bots */}
              <div style={{ display: "none" }} aria-hidden="true">
                <input
                  type="text"
                  tabIndex={-1}
                  autoComplete="off"
                  {...register("hp_field")}
                />
              </div>

              {/* Message input — Item #17 */}
              <div className="form-group">
                <label className="form-label" htmlFor="message-input">
                  Message Content: <span style={{ color: "var(--color-danger)" }} aria-label="required">*</span>
                </label>
                <textarea
                  id="message-input"
                  className={`form-textarea${errors.message ? " error" : ""}`}
                  placeholder="Paste an email body, subject, or SMS text here..."
                  style={{ minHeight: 200 }}
                  aria-describedby={errors.message ? "message-error" : "message-hint"}
                  aria-invalid={!!errors.message}
                  aria-required="true"
                  {...register("message", {
                    required: "Please enter a message to analyze.",
                    minLength: { value: MIN_LENGTH, message: `Message must be at least ${MIN_LENGTH} characters.` },
                    maxLength: { value: MAX_LENGTH, message: `Message cannot exceed ${MAX_LENGTH.toLocaleString()} characters.` },
                    validate: (v) => v.trim().length >= MIN_LENGTH || "Message must not be blank or whitespace only.",
                  })}
                />
                <div className={`char-count${charCount > MAX_LENGTH * 0.9 ? " near-limit" : ""}${charCount >= MAX_LENGTH ? " at-limit" : ""}`}>
                  {charCount.toLocaleString()} / {MAX_LENGTH.toLocaleString()}
                </div>
                {errors.message && (
                  <p id="message-error" className="form-error" role="alert">
                    ⚠ {errors.message.message}
                  </p>
                )}
                {!errors.message && (
                  <p id="message-hint" className="form-hint">
                    Supports email bodies, SMS messages, or any plain text. Content is never stored.
                  </p>
                )}
              </div>

              <div style={{ display: "flex", gap: "var(--space-sm)", flexWrap: "wrap" }}>
                <button
                  id="detect-btn"
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                  aria-label="Detect if this message is phishing"
                >
                  {loading ? (
                    <>
                      <span className="spinner" aria-hidden="true" style={{ width: 18, height: 18, borderWidth: 2 }}></span>
                      Analyzing…
                    </>
                  ) : "🚀 Detect Phishing"}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => { setValue("message", ""); setResult(null); setApiError(null); }}
                  aria-label="Clear message input"
                >
                  🧹 Clear
                </button>
              </div>
            </form>

            {/* API Error */}
            {apiError && (
              <div className="disclaimer mt-md" role="alert" aria-live="assertive">
                <strong>⚠ API Error:</strong> {apiError}
                <br />
                <small>Ensure the FastAPI backend is running: <code>uvicorn backend.main:app --reload</code></small>
              </div>
            )}

            {/* Results */}
            {result && (
              <div aria-live="polite" aria-label="Detection results">
                {result.prediction === 1 ? (
                  <div className="banner-phish" role="status">
                    <div className="banner-title" style={{ color: "var(--color-danger)" }}>
                      🚨 CLASSIFIED AS PHISHING
                    </div>
                    <p className="banner-sub">
                      Model evaluated high deceptive intent, social engineering keywords, or credential harvesting signals.
                    </p>
                  </div>
                ) : (
                  <div className="banner-safe" role="status">
                    <div className="banner-title" style={{ color: "var(--color-success)" }}>
                      🛡️ CLASSIFIED AS LEGITIMATE
                    </div>
                    <p className="banner-sub">
                      Model found typical conversational, transactional, or benign workplace linguistic structures.
                    </p>
                  </div>
                )}

                {/* Probability bar */}
                <div className="card mt-md">
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontWeight: 600 }}>
                      Phishing Risk Probability:{" "}
                      <span className="text-accent">{(result.phishing_probability * 100).toFixed(2)}%</span>
                    </span>
                    <span className="text-muted" style={{ fontSize: "0.88rem" }}>
                      Model: <code>{result.model_used}</code> | Confidence:{" "}
                      <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                    </span>
                  </div>
                  <div className="progress-bar" aria-label={`Phishing risk: ${(result.phishing_probability * 100).toFixed(1)}%`}>
                    <div
                      className="progress-fill"
                      style={{
                        width: `${result.phishing_probability * 100}%`,
                        background: result.prediction === 1
                          ? "linear-gradient(90deg, #ef4444, #b91c1c)"
                          : "linear-gradient(90deg, #10b981, #059669)",
                      }}
                    />
                  </div>
                </div>

                {/* Heuristic Grid */}
                <h2 className="section-title mt-lg">🔎 Rule-Based Security Heuristics</h2>
                <div className="grid-4">
                  <div className="indicator-card" aria-label={`URLs detected: ${result.heuristics.url_count}`}>
                    <div className="indicator-title">🔗 URLs Detected</div>
                    <span className="indicator-val" style={{ color: result.heuristics.url_count > 0 ? "var(--color-danger)" : "var(--color-success)" }}>
                      {result.heuristics.url_count}
                    </span>
                    <span className="indicator-sub">
                      {result.heuristics.has_ip_url ? "⚠️ IP Address in URL!" : "No IP-based URLs"}
                    </span>
                  </div>
                  <div className="indicator-card">
                    <div className="indicator-title">⏰ Urgency Keywords</div>
                    <span className="indicator-val" style={{ color: result.heuristics.urgency_triggers.length > 0 ? "var(--color-danger)" : "var(--color-success)" }}>
                      {result.heuristics.urgency_triggers.length}
                    </span>
                    <span className="indicator-sub">
                      {result.heuristics.urgency_triggers.slice(0, 3).join(", ") || "None detected"}
                    </span>
                  </div>
                  <div className="indicator-card">
                    <div className="indicator-title">💳 Credential Requests</div>
                    <span className="indicator-val" style={{ color: result.heuristics.credential_triggers.length > 0 ? "var(--color-danger)" : "var(--color-success)" }}>
                      {result.heuristics.credential_triggers.length}
                    </span>
                    <span className="indicator-sub">
                      {result.heuristics.credential_triggers.slice(0, 3).join(", ") || "None detected"}
                    </span>
                  </div>
                  <div className="indicator-card">
                    <div className="indicator-title">📊 Heuristic Risk Index</div>
                    <span className="indicator-val" style={{ color: result.heuristics.heuristic_risk_score > 50 ? "var(--color-danger)" : "var(--color-accent)" }}>
                      {result.heuristics.heuristic_risk_score} / 100
                    </span>
                    <span className="indicator-sub">Rule-based aggregate severity</span>
                  </div>
                </div>

                {/* Defanged URLs */}
                {result.heuristics.urls_detected.length > 0 && (
                  <div className="mt-md">
                    <h3 style={{ fontSize: "0.95rem", marginBottom: 8 }}>
                      Sanitized URL Preview (Defanged for Safety):
                    </h3>
                    {result.heuristics.urls_detected.map((url, i) => (
                      <code key={i} style={{ display: "block", background: "var(--color-surface)", padding: "8px 12px", borderRadius: 4, marginBottom: 4, fontSize: "0.85rem", wordBreak: "break-all" }}>
                        {url.replace(/https?:\/\//g, "hxxp://").replace(/\./g, "[.]")}
                      </code>
                    ))}
                  </div>
                )}

                {/* Token breakdown */}
                {result.influential_tokens.length > 0 && (
                  <>
                    <h2 className="section-title mt-lg">💡 Token Impact Breakdown (TF-IDF Log-Odds)</h2>
                    <p className="mb-md">
                      Words from this message that exerted the greatest mathematical pull on the classification decision:
                    </p>
                    <div style={{ overflowX: "auto" }}>
                      <table className="data-table" aria-label="Token impact breakdown">
                        <thead>
                          <tr>
                            <th scope="col">N-Gram Token</th>
                            <th scope="col">Pushes Toward</th>
                            <th scope="col">TF-IDF Score</th>
                            <th scope="col">Model Weight</th>
                            <th scope="col">Total Impact</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.influential_tokens.slice(0, 15).map((tok, i) => (
                            <tr key={i}>
                              <td><code>{tok.token}</code></td>
                              <td>
                                <span className={`badge ${tok.direction === "phishing" ? "badge-danger" : "badge-success"}`}>
                                  {tok.direction}
                                </span>
                              </td>
                              <td>{tok.tfidf.toFixed(4)}</td>
                              <td>{tok.model_weight.toFixed(4)}</td>
                              <td><strong>{tok.impact.toFixed(4)}</strong></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                {/* Disclaimer */}
                <div className="disclaimer mt-lg" role="note">
                  🛡️ <strong>Security Advisory:</strong> Supporting indicators and machine learning
                  classifications are statistical and heuristic estimations. They do not constitute
                  absolute proof of legitimacy or maliciousness. Never execute suspicious links or
                  submit sensitive credentials based solely on automated screening.
                </div>
              </div>
            )}
          </div>
        </div>

        {/* CTA — Item #20 */}
        <section className="cta-section" aria-labelledby="detector-cta">
          <h2 id="detector-cta">Compare Model Performance</h2>
          <p>
            See how all 4 classifiers compare side-by-side on training, validation, and test accuracy.
          </p>
          <Link href="/comparison" className="btn btn-primary">
            ⚖️ View Model Comparison →
          </Link>
        </section>

      </div>
    </div>
  );
}
