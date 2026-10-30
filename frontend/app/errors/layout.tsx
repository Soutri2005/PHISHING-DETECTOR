import type { Metadata } from "next";
export const metadata: Metadata = {
  title: "Error Analysis & Explainability",
  description: "Diagnose false positives and false negatives in phishing detection. Explore token-level TF-IDF log-odds weights and known attack vector vulnerabilities.",
};
export default function ErrorsLayout({ children }: { children: React.ReactNode }) { return children; }
