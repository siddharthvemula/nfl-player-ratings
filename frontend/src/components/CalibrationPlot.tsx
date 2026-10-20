import { useState } from "react";
import type { Meta } from "../api";
import { signed } from "../format";
import { ticks, useWidth } from "./QbLandscape";

type Bin = Meta["test_2025"]["calibration"][number];
const M = { top: 16, right: 20, bottom: 48, left: 54 };

/**
 * Calibration: games sorted by forecast and cut into 10 equal groups. x = what the model said (group average),
 * y = what actually happened. On the dashed diagonal the forecast was right on average; above it the model was
 * too pessimistic, below it too optimistic. Whiskers are ±1 standard error: how far luck alone could move a dot.
 */
export function CalibrationPlot({ bins }: { bins: Bin[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const height = Math.round(Math.min(460, Math.max(300, width * 0.62)));
  const lo = Math.min(...bins.flatMap((b) => [b.predicted, b.actual - b.se])) - 0.02;
  const hi = Math.max(...bins.flatMap((b) => [b.predicted, b.actual + b.se])) + 0.02;
  // Same range on both axes, so the diagonal really is 45° and "above/below" is honest.
  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const sx = (v: number) => M.left + ((v - lo) / (hi - lo)) * iw;
  const sy = (v: number) => M.top + (1 - (v - lo) / (hi - lo)) * ih;
  const t = ticks(lo, hi, width < 560 ? 4 : 6);
  const top = bins[bins.length - 1];
  const active = hover === null ? null : bins[hover];

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={width} height={height} role="img" aria-label="Calibration plot: average forecast against average actual EPA per dropback, ten groups of 2025 games">
        {t.map((v) => (
          <g key={v}>
            <line x1={M.left} x2={width - M.right} y1={sy(v)} y2={sy(v)} stroke="var(--grid)" />
            <line x1={sx(v)} x2={sx(v)} y1={M.top} y2={height - M.bottom} stroke="var(--grid)" />
            <text x={M.left - 8} y={sy(v) + 4} textAnchor="end" fontSize="11.5" fill="var(--text-muted)">{signed(v, 2)}</text>
            <text x={sx(v)} y={height - M.bottom + 17} textAnchor="middle" fontSize="11.5" fill="var(--text-muted)">{signed(v, 2)}</text>
          </g>
        ))}
        <line x1={M.left} x2={M.left} y1={M.top} y2={height - M.bottom} stroke="var(--axis)" />
        <line x1={M.left} x2={width - M.right} y1={height - M.bottom} y2={height - M.bottom} stroke="var(--axis)" />
        <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="5 4" />
        <text x={sx(lo) + 8} y={sy(lo) - 8} fontSize="11.5" fill="var(--text-muted)" transform={`rotate(-45 ${sx(lo) + 8} ${sy(lo) - 8})`}>perfect calibration</text>
        <text x={M.left + iw / 2} y={height - 8} textAnchor="middle" fontSize="12" fill="var(--text-secondary)">Average forecast in the group (EPA/dropback)</text>
        <text x={14} y={M.top + ih / 2} textAnchor="middle" fontSize="12" fill="var(--text-secondary)" transform={`rotate(-90 14 ${M.top + ih / 2})`}>Average actual</text>
        {bins.map((b, i) => (
          <g key={b.bin}>
            <line x1={sx(b.predicted)} x2={sx(b.predicted)} y1={sy(b.actual - b.se)} y2={sy(b.actual + b.se)} stroke="var(--series-actual)" strokeWidth="2" opacity="0.55" />
            <circle cx={sx(b.predicted)} cy={sy(b.actual)} r={hover === i ? 7 : 5.5} fill="var(--series-actual)" stroke="var(--surface-1)" strokeWidth="2" />
            <circle cx={sx(b.predicted)} cy={sy(b.actual)} r={16} fill="transparent" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onClick={() => setHover(hover === i ? null : i)} />
          </g>
        ))}
        <text x={width - M.right} y={sy(top.actual - top.se) + 22} textAnchor="end" fontSize="12" fill="var(--text-secondary)">
          <tspan x={width - M.right}>Top group</tspan>
          <tspan x={width - M.right} dy="15">said {signed(top.predicted)}, got {signed(top.actual)}</tspan>
        </text>
      </svg>
      {active && (
        <div className="tooltip" style={{ left: Math.min(Math.max(sx(active.predicted) + 14, 8), width - 200), top: Math.max(sy(active.actual) - 70, 0) }}>
          <div className="t-head">Group {active.bin} of {bins.length} · {active.games} games</div>
          <div className="t-row"><span>Model said</span><b>{signed(active.predicted)}</b></div>
          <div className="t-row"><span>Actually</span><b>{signed(active.actual)} <span className="muted">± {active.se.toFixed(3)}</span></b></div>
        </div>
      )}
    </div>
  );
}
