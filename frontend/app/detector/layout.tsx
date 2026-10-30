import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Phishing Detector — Live Inference",
  description:
    "Test any email or SMS message with our live TF-IDF + ML pipeline. Get real-time phishing probability, heuristic breakdown, and token-level explanations.",
};

export default function DetectorLayout({ children }: { children: React.ReactNode }) {
  return children;
}
