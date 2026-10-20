import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import type { Rating } from "../api";
import { signed } from "../format";

interface Point {
  key: string;
  label: string;
  rating: number;
  actual: number | null;
  game: Rating;
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: { payload: Point }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tooltip">
      <div className="t-head">{p.game.season} · Week {p.game.week}{p.game.opponent ? ` vs ${p.game.opponent}` : ""}</div>
      <div className="t-row"><span><span className="legend"><span className="key" style={{ background: "var(--series-rating)", marginRight: 6 }} /></span>Rating before</span><b>{signed(p.rating)}</b></div>
      <div className="t-row"><span><span className="legend"><span className="dot" style={{ background: "var(--series-actual)", marginRight: 6 }} /></span>Actual</span><b>{p.actual === null ? "not played yet" : signed(p.actual)}</b></div>
      {p.game.dropbacks !== null && <div className="t-row">Dropbacks <b>{p.game.dropbacks}</b></div>}
      <div className="muted small" style={{ marginTop: 6 }}>Click to see why</div>
    </div>
  );
}

/**
 * The rating made before each game (line) and what he actually did (dots); the upcoming game is a hollow dot.
 * Click any game to break its rating down below.
 */
export function RatingChart({ games, selected, onSelect, leagueAvg }: { games: Rating[]; selected: Rating | null; onSelect: (g: Rating) => void; leagueAvg: number }) {
  const data: Point[] = games.map((g) => ({
    key: g.game_id + g.passer_id,
    label: `${String(g.season).slice(2)}·W${g.week}`,
    rating: g.matchup_rating,
    actual: g.actual,
    game: g,
  }));
  const select = (state: { activePayload?: { payload: Point }[] } | null) => {
    const p = state?.activePayload?.[0]?.payload;
    if (p) onSelect(p.game);
  };
  const tick = { fill: "var(--text-muted)", fontSize: 11 };
  return (
    <>
      <div className="legend" aria-hidden="true">
        <span><span className="key" style={{ background: "var(--series-rating)" }} />Rating (made before the game)</span>
        <span><span className="dot" style={{ background: "var(--series-actual)" }} />Actual EPA/dropback</span>
        <span><span className="dot" style={{ border: "2px solid var(--series-rating)" }} />Upcoming</span>
        <span><span className="key dashed" />League average</span>
      </div>
      <div style={{ width: "100%", height: 300 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -6 }} onClick={select} style={{ cursor: "pointer" }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="label" tick={tick} tickLine={false} axisLine={{ stroke: "var(--axis)" }} minTickGap={28} />
            <YAxis tick={tick} tickLine={false} axisLine={false} tickFormatter={(v: number) => signed(v, 1)} width={48} />
            <ReferenceLine y={leagueAvg} stroke="var(--text-muted)" strokeDasharray="3 4" />
            {selected && <ReferenceLine x={data.find((d) => d.game.game_id === selected.game_id)?.label} stroke="var(--accent)" strokeOpacity={0.22} strokeWidth={14} />}
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--text-muted)", strokeWidth: 1 }} wrapperStyle={{ outline: "none" }} />
            <Scatter dataKey="actual" fill="var(--series-actual)" isAnimationActive={false} shape={(props: { cx?: number; cy?: number; payload?: Point }) =>
              // Unplayed games have no actual yet: draw nothing (recharts would otherwise pin them to the top edge).
              props.cx === undefined || props.cy === undefined || props.payload?.actual == null
                ? <g />
                : <circle cx={props.cx} cy={props.cy} r={4.5} fill="var(--series-actual)" stroke="var(--surface-1)" strokeWidth={2} />} />
            <Line dataKey="rating" stroke="var(--series-rating)" strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" animationDuration={600}
              dot={(props: { cx?: number; cy?: number; payload?: Point; index?: number }) =>
                props.payload?.game.upcoming
                  ? <circle key={props.index} cx={props.cx} cy={props.cy} r={6} fill="var(--surface-1)" stroke="var(--series-rating)" strokeWidth={2.5} />
                  : <g key={props.index} />}
              activeDot={{ r: 5.5, stroke: "var(--surface-1)", strokeWidth: 2 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
