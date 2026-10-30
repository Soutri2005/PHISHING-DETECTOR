import type { Metadata } from "next";
export const metadata: Metadata = {
  title: "Model Comparison — Benchmark",
  description: "Side-by-side benchmark of Logistic Regression, Naive Bayes, Decision Tree, and Random Forest classifiers. Confusion matrices, generalization gaps, and test-set verification.",
};
export default function ComparisonLayout({ children }: { children: React.ReactNode }) {
  return children;
}
