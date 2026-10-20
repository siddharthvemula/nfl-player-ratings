import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import type { Rating } from "../api";
import { signed, weekLabel } from "../format";

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
      <div className="secondary">
        {weekLabel(p.game.season, p.game.week)} {p.game.opponent ? `vs ${p.game.opponent}` : ""}
      </div>
      <div>
        <span className="legend"><span className="key" style={{ background: "var(--series-rating)" }} /></span>
        <span className="v">{signed(p.rating)}</span> <span className="muted">rating</span>
      </div>
      <div>
        <span className="legend"><span className="dot" style={{ background: "var(--series-actual)" }} /></span>
        <span className="v">{p.actual === null ? "not played yet" : signed(p.actual)}</span> <span className="muted">actual</span>
      </div>
    </div>
  );
}

/** Rating made before each game (line) vs what he actually did (dots). Click a game to see its breakdown. */
export function RatingChart({ games, onSelect }: { games: Rating[]; onSelect: (g: Rating) => void }) {
  const data: Point[] = games.map((g) => ({
    key: g.game_id + g.passer_id,
    label: `${String(g.season).slice(2)}·${g.week}`,
    rating: g.matchup_rating,
    actual: g.actual,
    game: g,
  }));
  const select = (state: { activePayload?: { payload: Point }[] } | null) => {
    const p = state?.activePayload?.[0]?.payload;
    if (p) onSelect(p.game);
  };
  return (
    <>
      <div className="legend" aria-hidden="true">
        <span><span className="key" style={{ background: "var(--series-rating)" }} />Rating (before the game)</span>
        <span><span className="dot" style={{ background: "var(--series-actual)" }} />Actual EPA/dropback</span>
      </div>
      <div style={{ width: "100%", height: 280 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -8 }} onClick={select}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--grid)" }} minTickGap={24} />
            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={(v: number) => v.toFixed(2)} width={48} />
            <ReferenceLine y={0} stroke="var(--text-muted)" />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--text-muted)", strokeWidth: 1 }} />
            <Scatter dataKey="actual" fill="var(--series-actual)" isAnimationActive={false} shape={(props: { cx?: number; cy?: number; payload?: Point }) =>
              // Unplayed games have no actual yet: draw nothing (recharts would otherwise pin them to the top edge).
              props.cx === undefined || props.cy === undefined || props.payload?.actual == null
                ? <g />
                : <circle cx={props.cx} cy={props.cy} r={5} fill="var(--series-actual)" stroke="var(--surface-1)" strokeWidth={2} />} />
            <Line dataKey="rating" stroke="var(--series-rating)" strokeWidth={2} dot={false} activeDot={{ r: 5, stroke: "var(--surface-1)", strokeWidth: 2 }} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
