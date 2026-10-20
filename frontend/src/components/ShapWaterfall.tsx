import type { Rating } from "../api";
import { FEATURE_LABELS, featureValue, signed } from "../format";

interface Step {
  name: string;
  detail?: string;
  value: number; // this step's own amount
  from: number; // running total before it
  to: number; // running total after it
  kind: "total" | "up" | "down";
}

/**
 * How one rating is built: carry-over, then the average adjustment, then one piece per feature
 * (weight × (his value − training average)), landing exactly on the matchup rating.
 */
export function ShapWaterfall({ game, order }: { game: Rating; order: string[] }) {
  const steps: Step[] = [{ name: "Carry-over", detail: "his shrunk EPA/dropback so far", value: game.carry_over, from: 0, to: game.carry_over, kind: "total" }];
  let run = game.carry_over;
  const push = (name: string, value: number, detail?: string) => {
    steps.push({ name, detail, value, from: run, to: run + value, kind: value >= 0 ? "up" : "down" });
    run += value;
  };
  push("Average adjustment", game.avg_adjustment, "for a typical training game");
  // Biggest pieces first, so the eye lands on what matters.
  [...order]
    .sort((a, b) => Math.abs(game.contributions[b]) - Math.abs(game.contributions[a]))
    .forEach((f) => push(FEATURE_LABELS[f] ?? f, game.contributions[f], featureValue(f, game.features[f])));
  const opp = game.opponent ? `vs ${game.opponent}` : "neutral opponent";
  steps.push({ name: "Rating", detail: opp, value: run, from: 0, to: run, kind: "total" });

  // One shared scale that includes 0 and every running total.
  const xs = steps.flatMap((s) => [s.from, s.to]).concat(0);
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const pad = (hi - lo) * 0.05 || 0.01;
  const pct = (v: number) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * 100;

  return (
    <div className="wf" role="table" aria-label="How this rating is built">
      {steps.map((s, i) => {
        const left = pct(Math.min(s.from, s.to));
        const width = Math.abs(pct(s.to) - pct(s.from));
        const isTotal = i === steps.length - 1;
        return (
          <div key={s.name} role="row" style={{ display: "contents" }}>
            <div role="cell" className={`name ${isTotal ? "total-row" : ""}`}>
              {s.name}
              {s.detail && <small>{s.detail}</small>}
            </div>
            <div role="cell" className="track" title={`${s.name}: ${signed(s.value)} (running total ${signed(s.to)})`}>
              <span className="zero" style={{ left: `${pct(0)}%` }} />
              <span className={`bar ${s.kind}`} style={{ left: `${left}%`, width: `${width}%` }} />
            </div>
            <div role="cell" className={`num ${isTotal ? "total-row" : ""}`}>
              {signed(s.value)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
