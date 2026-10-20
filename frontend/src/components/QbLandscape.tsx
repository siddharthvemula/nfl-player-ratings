import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { Rating, Rankings } from "../api";
import { signed } from "../format";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { TeamChip, useTeamColor } from "./Identity";

const M = { top: 30, right: 20, bottom: 46, left: 50 };
const MIN_R = 4; // >= 8px markers
const MAX_R = 15;
const SOLID_EVIDENCE = 30; // QBs with fewer dropbacks this season don't stretch the x-axis (they're pinned to its edge)

/** Measure an element's width (charts redraw to fit; no fixed pixel sizes). */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    if (!ref.current) return;
    setWidth(Math.round(ref.current.clientWidth) || 720); // measure now; the observer below keeps it current
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Round tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, target = 6) {
  const raw = (hi - lo) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

/** Weighted least squares line y = a + b x (weights = dropbacks, as in the rest of the project). */
export function weightedFit(pts: { x: number; y: number; w: number }[]) {
  const W = pts.reduce((s, p) => s + p.w, 0);
  if (pts.length < 3 || W === 0) return null;
  const mx = pts.reduce((s, p) => s + p.w * p.x, 0) / W;
  const my = pts.reduce((s, p) => s + p.w * p.y, 0) / W;
  const sxx = pts.reduce((s, p) => s + p.w * (p.x - mx) ** 2, 0);
  if (sxx < 1e-12) return null;
  const b = pts.reduce((s, p) => s + p.w * (p.x - mx) * (p.y - my), 0) / sxx;
  return { a: my - b * mx, b };
}

// The x value: his season-to-date raw EPA/dropback. Before his first dropback that's the model's own placeholder,
// the league average (features.std_avg), so week 1 starts everyone on one vertical line.
const xOf = (r: Rating) => r.raw_epa ?? r.features.std_avg;

export function QbLandscape({ weeks, index, onIndex, leagueAvg, seasonLabel }: {
  weeks: Rankings[]; index: number; onIndex: (i: number) => void; leagueAvg: number; seasonLabel: string;
}) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const navigate = useNavigate();
  const teamColor = useTeamColor();
  const [hover, setHover] = useState<string | null>(null);
  const [pinned, setPinned] = useState<string | null>(null); // touch: first tap shows the tooltip
  const [playing, setPlaying] = useState(false);
  const narrow = width < 560;
  const height = Math.round(Math.min(500, Math.max(320, width * 0.56)));
  const week = weeks[index];

  // Axes and dot sizes are fixed for the whole season, so motion between weeks means something.
  const scales = useMemo(() => {
    const all = weeks.flatMap((w) => w.rows);
    const solid = all.filter((r) => r.season_dropbacks >= SOLID_EVIDENCE).map(xOf).concat(leagueAvg);
    const ys = all.map((r) => r.rating);
    const [x0, x1] = [Math.min(...solid), Math.max(...solid)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const px = (x1 - x0) * 0.06 || 0.1;
    const py = (y1 - y0) * 0.08 || 0.05;
    return {
      xd: [x0 - px, x1 + px] as const,
      yd: [y0 - py, y1 + py] as const,
      maxDb: Math.max(1, ...all.map((r) => r.season_dropbacks)),
    };
  }, [weeks, leagueAvg]);

  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const sx = (v: number) => M.left + ((Math.min(Math.max(v, scales.xd[0]), scales.xd[1]) - scales.xd[0]) / (scales.xd[1] - scales.xd[0])) * iw;
  const sy = (v: number) => M.top + (1 - (v - scales.yd[0]) / (scales.yd[1] - scales.yd[0])) * ih;
  // Square-root scale: a dot's AREA grows with dropbacks, so a 20-dropback backup reads as a small sample.
  const sr = (db: number) => MIN_R + (MAX_R - MIN_R) * Math.sqrt(db / scales.maxDb);

  const rows = week?.rows ?? [];
  const fit = useMemo(() => weightedFit(rows.filter((r) => r.season_dropbacks > 0).map((r) => ({ x: xOf(r), y: r.rating, w: r.season_dropbacks }))), [rows]);

  // Direct labels for the top QBs, nudged so they don't collide.
  const labels = useMemo(() => {
    const top = [...rows].sort((a, b) => a.rank - b.rank).slice(0, narrow ? 3 : 5);
    // Every dot is an obstacle, so labels never sit on top of another QB.
    const placed: { x: number; y: number; w: number; h: number }[] = rows.map((r) => {
      const rad = sr(r.season_dropbacks);
      return { x: sx(xOf(r)) - rad, y: sy(r.rating) - rad, w: 2 * rad, h: 2 * rad };
    });
    const out: { id: string; x: number; y: number; anchor: "start" | "end"; text: string }[] = [];
    for (const r of top) {
      const cx = sx(xOf(r));
      const cy = sy(r.rating);
      const rad = sr(r.season_dropbacks);
      const text = narrow ? r.name.split(" ").slice(-1)[0] : r.name;
      const w = text.length * 6.6 + 4;
      const own = { x: cx - rad, y: cy - rad, w: 2 * rad, h: 2 * rad };
      const others = placed.filter((p) => !(p.x === own.x && p.y === own.y));
      const options = [
        { x: cx + rad + 5, y: cy + 4, anchor: "start" as const },
        { x: cx - rad - 5, y: cy + 4, anchor: "end" as const },
        { x: cx + rad + 3, y: cy - rad - 3, anchor: "start" as const },
        { x: cx + rad + 3, y: cy + rad + 12, anchor: "start" as const },
        { x: cx - rad - 3, y: cy - rad - 3, anchor: "end" as const },
        { x: cx - rad - 3, y: cy + rad + 12, anchor: "end" as const },
      ];
      const box = (o: (typeof options)[number]) => ({ x: o.anchor === "start" ? o.x : o.x - w, y: o.y - 11, w, h: 14 });
      const fits = (b: ReturnType<typeof box>) => b.x >= 0 && b.x + b.w <= width && !others.some((p) => b.x < p.x + p.w && b.x + b.w > p.x && b.y < p.y + p.h && b.y + b.h > p.y);
      const o = options.find((o) => fits(box(o))) ?? options[0];
      placed.push(box(o));
      others.push(box(o));
      out.push({ id: r.passer_id, x: o.x, y: o.y, anchor: o.anchor, text });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, width, narrow, scales]);

  // Play: step through the weeks.
  useEffect(() => {
    if (!playing) return;
    if (index >= weeks.length - 1) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => onIndex(index + 1), 1100);
    return () => clearTimeout(t);
  }, [playing, index, weeks.length, onIndex]);

  const activeId = pinned ?? hover;
  const active = rows.find((r) => r.passer_id === activeId);
  const xt = ticks(scales.xd[0], scales.xd[1], narrow ? 4 : 7);
  const yt = ticks(scales.yd[0], scales.yd[1], 5);
  const firstWeek = rows.every((r) => r.season_dropbacks === 0);
  const drawOrder = [...rows].sort((a, b) => b.season_dropbacks - a.season_dropbacks);

  return (
    <div>
      <div className="slider" style={{ marginBottom: 12 }}>
        <button className="play" aria-label={playing ? "Pause" : "Play through the weeks"} onClick={() => {
          if (!playing && index >= weeks.length - 1) onIndex(0);
          setPlaying((p) => !p);
        }}>
          {playing ? "❚❚" : "▶"}
        </button>
        <span className="week-now">Week {week?.week}{week?.upcoming ? " ·  next" : ""}</span>
        <input type="range" min={0} max={weeks.length - 1} value={index} aria-label={`Week of ${seasonLabel}`}
          onChange={(e) => { setPlaying(false); onIndex(Number(e.target.value)); }} />
        <span className="muted small">Wk {weeks[0]?.week}–{weeks[weeks.length - 1]?.week}</span>
      </div>

      <div className="chart-wrap" ref={wrapRef} onPointerLeave={() => setHover(null)}>
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
          aria-label={`Scatter of ${rows.length} QBs: raw EPA per dropback this season against the model's rating. The table below lists every value.`}
          onClick={(e) => { if (e.target === e.currentTarget) setPinned(null); }}>
          {yt.map((v) => (
            <g key={`y${v}`} className="axis">
              <line className="gridline" x1={M.left} x2={width - M.right} y1={sy(v)} y2={sy(v)} />
              <text x={M.left - 8} y={sy(v) + 4} textAnchor="end">{signed(v, 2)}</text>
            </g>
          ))}
          {xt.map((v) => (
            <g key={`x${v}`} className="axis">
              <line className="gridline" x1={sx(v)} x2={sx(v)} y1={M.top} y2={height - M.bottom} />
              <text x={sx(v)} y={height - M.bottom + 16} textAnchor="middle">{signed(v, 1)}</text>
            </g>
          ))}
          {/* League average on both axes: the "average QB" crosshair. */}
          <line className="zeroline" x1={sx(leagueAvg)} x2={sx(leagueAvg)} y1={M.top} y2={height - M.bottom} strokeDasharray="2 4" />
          <line className="zeroline" x1={M.left} x2={width - M.right} y1={sy(leagueAvg)} y2={sy(leagueAvg)} strokeDasharray="2 4" />
          <text className="axis-title" x={M.left + iw / 2} y={height - 6} textAnchor="middle">Raw EPA per dropback this season →</text>
          <text className="axis-title" x={M.left - 8} y={M.top - 14}>↑ Model rating</text>

          <defs>
            <clipPath id="plot-area"><rect x={M.left} y={M.top} width={iw} height={ih} /></clipPath>
          </defs>
          {fit && (
            <line className="fit-line" clipPath="url(#plot-area)" stroke="var(--text-secondary)" strokeWidth={2} strokeLinecap="round" opacity={0.75}
              x1={sx(scales.xd[0])} y1={sy(fit.a + fit.b * scales.xd[0])} x2={sx(scales.xd[1])} y2={sy(fit.a + fit.b * scales.xd[1])} />
          )}

          {drawOrder.map((r) => {
            const x = sx(xOf(r));
            const y = sy(r.rating);
            const rad = sr(r.season_dropbacks);
            const clipped = xOf(r) < scales.xd[0] || xOf(r) > scales.xd[1];
            return (
              <g key={r.passer_id} className={`dot-g ${activeId === r.passer_id ? "active" : ""}`} style={{ transform: `translate(${x}px, ${y}px)` }}
                onPointerEnter={(e) => e.pointerType === "mouse" && setHover(r.passer_id)}
                onPointerUp={(e) => {
                  if (e.pointerType === "mouse") navigate(`/player/${r.passer_id}`);
                  else setPinned((p) => (p === r.passer_id ? null : r.passer_id));
                }}>
                <circle r={rad + 6} fill="transparent" />
                <circle className="mark" r={rad} fill={teamColor(r.team)} stroke="var(--surface-1)" strokeWidth={2} fillOpacity={0.92} />
                {clipped && <text y={-rad - 3} textAnchor="middle" className="tick">{xOf(r) < scales.xd[0] ? "◂" : "▸"}</text>}
              </g>
            );
          })}

          {labels.map((l) => (
            <text key={l.id} className="dot-label" x={l.x} y={l.y} textAnchor={l.anchor}>{l.text}</text>
          ))}
        </svg>

        {active && (() => {
          const x = sx(xOf(active));
          const y = sy(active.rating);
          const left = x + 16 + 210 > width ? Math.max(0, x - 226) : x + 16;
          const top = Math.min(Math.max(0, y - 40), height - 170);
          return (
            <div className={`tooltip ${pinned ? "interactive" : ""}`} style={{ left, top }} role="status">
              <div className="t-head">{active.name} <TeamChip team={active.team} /></div>
              <div className="t-row">Rating <b>{signed(active.rating)}</b></div>
              <div className="t-row">Raw EPA/dropback <b>{active.raw_epa === null ? "no games yet" : signed(active.raw_epa)}</b></div>
              <div className="t-row">Dropbacks this season <b>{active.season_dropbacks}</b></div>
              <div className="t-row">Rank <b>#{active.rank}</b></div>
              <div className="t-row" style={{ marginTop: 4 }}>Confidence <ConfidenceBadge level={active.confidence} evidence={active.evidence} /></div>
              {pinned ? <Link className="t-link" to={`/player/${active.passer_id}`}>Open player →</Link> : <div className="muted small" style={{ marginTop: 6 }}>Click to open his page</div>}
            </div>
          );
        })()}
      </div>

      <div className="legend" style={{ marginTop: 10 }}>
        <span><span className="dot" style={{ background: "var(--text-muted)" }} />One QB, in his team's color · bigger = more dropbacks this season</span>
        <span><span className="key" style={{ background: "var(--text-secondary)" }} />Fitted line (weighted by dropbacks)</span>
        <span><span className="key dashed" />League average</span>
      </div>
      <p className="chart-caption">
        {firstWeek ? (
          <>Week 1: nobody has played yet, so every QB sits at the league average ({signed(leagueAvg, 3)}) and the ratings come from last season.
            Press play to watch them spread out.</>
        ) : (
          <><strong>Above the line</strong>: the model rates him higher than his raw stats. <strong>Below the line</strong>: a hot start the
            model is skeptical of.{fit && <> The line's slope is <strong>{fit.b.toFixed(2)}</strong>: each +0.10 of raw EPA this season moves
            the rating about {signed(fit.b * 0.1, 2)}, because a few games are mostly noise.</>}</>
        )}
      </p>
    </div>
  );
}
