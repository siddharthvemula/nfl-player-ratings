import type { Confidence } from "../api";

const BARS: Record<Confidence, number> = { Low: 1, Med: 2, High: 3 };

/** Confidence = dropbacks of evidence (last season + this season). Text label + a 3-step meter, never color alone. */
export function ConfidenceBadge({ level, evidence }: { level: Confidence; evidence?: number }) {
  const title = evidence === undefined ? `${level} confidence` : `${level} confidence: ${evidence} dropbacks of evidence`;
  return (
    <span className="badge" title={title}>
      <span className="meter" aria-hidden="true">
        {[1, 2, 3].map((i) => (
          <i key={i} className={i <= BARS[level] ? "on" : ""} />
        ))}
      </span>
      {level}
    </span>
  );
}
