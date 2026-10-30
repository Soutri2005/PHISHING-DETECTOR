/**
 * Root Layout — applies to every page.
 * Items addressed:
 *   #6  — Meta titles & descriptions via generateMetadata
 *   #7  — Social preview OG image
 *   #8  — Favicon links
 *   #5  — Cookie consent banner (loaded client-side)
 *   #12 — Google font via next/font (no layout shift, self-hosted)
 *   #14 — Viewport meta (mobile-friendly)
 *   #19 — Analytics script (conditionally loaded after consent)
 */
import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { NavBar } from "@/components/NavBar";
import { Footer } from "@/components/Footer";
import { CookieBanner } from "@/components/CookieBanner";
import { AnalyticsScript } from "@/components/AnalyticsScript";

// Item #12 — Subset font, self-hosted by Next.js (no FOUT)
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL || "https://phishing-detector.example.com";

// Item #6 — Default metadata (each page overrides title & description)
export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Phishing Detector — Overfitting Study",
    template: "%s | Phishing Detector",
  },
  description:
    "An end-to-end ML cybersecurity application evaluating email text classification, overfitting, and regularization techniques on 17,522 real phishing emails.",
  keywords: [
    "phishing detector",
    "machine learning",
    "overfitting",
    "cybersecurity",
    "NLP",
    "TF-IDF",
    "logistic regression",
  ],
  authors: [{ name: "Phishing Detector Study" }],
  // Item #7 — Social preview image (OG + Twitter Card)
  openGraph: {
    type: "website",
    locale: "en_US",
    url: SITE_URL,
    siteName: "Phishing Detector — Overfitting Study",
    title: "Phishing Detector — Overfitting Study",
    description:
      "End-to-end ML cybersecurity research: phishing classification, overfitting analysis, and regularization across 4 classifiers.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Phishing Detector Overfitting Study — cybersecurity ML dashboard showing 97.3% test accuracy",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Phishing Detector — Overfitting Study",
    description:
      "End-to-end ML: phishing classification, overfitting analysis, 4 classifiers, 17,522 emails.",
    images: ["/og-image.png"],
  },
  // Item #8 — Favicon
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon-32x32.png", type: "image/png", sizes: "32x32" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
  },
  manifest: "/site.webmanifest",
  robots: {
    index: true,
    follow: true,
  },
};

// Item #14 — Mobile friendly viewport
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0b1120",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        {/* Item #19 — GA4 (loaded only after cookie consent) */}
        <AnalyticsScript />

        <NavBar />

        <main id="main-content" tabIndex={-1}>
          {children}
        </main>

        <Footer />

        {/* Item #5 — GDPR cookie consent banner */}
        <CookieBanner />
      </body>
    </html>
  );
}
