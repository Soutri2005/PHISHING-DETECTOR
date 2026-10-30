import type { Metadata } from "next";
export const metadata: Metadata = {
  title: "Overfitting Lab — Interactive Experiments",
  description: "Interactively explore how Decision Tree depth and Logistic Regression regularization strength affect the generalization gap on phishing email data.",
};
export default function LabLayout({ children }: { children: React.ReactNode }) { return children; }
