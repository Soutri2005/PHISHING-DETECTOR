/**
 * Footer Component
 * Items: #16 (no broken links — all internal), #1/#2 (links to privacy/terms)
 */
import Link from "next/link";

export function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer role="contentinfo">
      <div className="footer-inner">
        <p className="footer-copy">
          © {year} Phishing Detector — Overfitting Study. For research &amp; educational use only.
        </p>
        <nav className="footer-links" aria-label="Footer navigation">
          <Link href="/privacy">Privacy Policy</Link>
          <Link href="/terms">Terms &amp; Conditions</Link>
          <Link href="/methodology">Methodology</Link>
          <a
            href="https://huggingface.co/datasets/zefang-liu/phishing-email-dataset"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="HuggingFace dataset (opens in new tab)"
          >
            Dataset
          </a>
        </nav>
      </div>
    </footer>
  );
}
