import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useApi, type Meta, type Player as PlayerT, type Rating } from "../api";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { RatingChart } from "../components/RatingChart";
import { ShapWaterfall } from "../components/ShapWaterfall";
import { signed, weekLabel } from "../format";

export function Player() {
  const { id } = useParams();
  const { data, error, loading } = useApi<PlayerT>(id ? `/api/players/${id}` : null);
  const meta = useApi<Meta>("/api/meta");
  const seasons = useMemo(() => [...new Set(data?.games.map((g) => g.season) ?? [])].reverse(), [data]);
  const [season, setSeason] = useState<number | "all">("all");
  const [selected, setSelected] = useState<Rating | null>(null);

  // Default view: his latest season, and his latest game (the upcoming one if he has it).
  useEffect(() => {
    if (!data) return;
    setSeason(seasons[0] ?? "all");
    setSelected(data.games[data.games.length - 1]);
  }, [data, seasons]);

  if (error) return <p className="error">{error}</p>;
  if (loading || !data) return <p className="muted">Loading…</p>;

  const games = season === "all" ? data.games : data.games.filter((g) => g.season === season);
  const latest = data.games[data.games.length - 1];
  const played = data.games.filter((g) => g.actual !== null);
  const features = meta.data?.features ?? Object.keys(latest.contributions);

  return (
    <>
      <p><Link to="/">← Rankings</Link></p>
      <h1>{data.passer} <span className="muted">{data.team}</span></h1>

      <div className="cards">
        <div className="card stat">
          <div className="label">{latest.upcoming ? `Rating for ${weekLabel(latest.season, latest.week)}` : `Last rating (${weekLabel(latest.season, latest.week)})`}</div>
          <div className="value">{signed(latest.rating)}</div>
          <div className="sub">{latest.upcoming ? `#${latest.rank} this week · ` : ""}EPA/dropback, average defense</div>
        </div>
        <div className="card stat">
          <div className="label">{latest.opponent ? `Next: vs ${latest.opponent}` : "Next opponent"}</div>
          <div className="value">{latest.opponent ? signed(latest.matchup_rating) : "—"}</div>
          <div className="sub">forecast against the actual defense</div>
        </div>
        <div className="card stat">
          <div className="label">Confidence</div>
          <div className="value"><ConfidenceBadge level={latest.confidence} evidence={latest.evidence} /></div>
          <div className="sub">{latest.evidence} dropbacks of evidence</div>
        </div>
        <div className="card stat">
          <div className="label">Rated games since 2019</div>
          <div className="value">{played.length}</div>
          <div className="sub">{played.length ? `${played[0].season}-${played[played.length - 1].season}` : "none yet"}</div>
        </div>
      </div>

      <h2>Rating history</h2>
      <div className="controls">
        <select value={season} onChange={(e) => setSeason(e.target.value === "all" ? "all" : Number(e.target.value))} aria-label="Season">
          {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
          <option value="all">All seasons</option>
        </select>
        <span className="muted">Click a game to see why it got its rating.</span>
      </div>
      <div className="card">
        <RatingChart games={games} onSelect={setSelected} />
      </div>

      {selected && (
        <>
          <h2>Why {signed(selected.matchup_rating)}? <span className="muted">{weekLabel(selected.season, selected.week)}{selected.opponent ? ` vs ${selected.opponent}` : ""}</span></h2>
          <p className="secondary">
            Every rating starts from <strong>carry-over</strong> (his EPA/dropback, last season blended into this one and pulled toward
            the league average when there's little to go on). The model then adjusts it: each bar is one feature's push
            (weight × how far his value is from a typical game), and the pieces add up exactly to the rating.
            {selected.actual !== null && <> He actually posted <strong>{signed(selected.actual)}</strong>.</>}
          </p>
          <div className="card"><ShapWaterfall game={selected} order={features} /></div>
          <p className="legend">
            <span><span className="dot" style={{ background: "var(--up)", borderRadius: 2 }} />raises the rating</span>
            <span><span className="dot" style={{ background: "var(--down)", borderRadius: 2 }} />lowers it</span>
            <span><span className="dot" style={{ background: "var(--total)", borderRadius: 2 }} />starting point / result</span>
          </p>
        </>
      )}

      <h2>Games</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Week</th><th>Opp</th><th className="num">Rating</th><th className="num">vs opp</th>
              <th className="num">Actual</th><th className="num">Dropbacks</th><th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {[...games].reverse().map((g) => (
              <tr key={g.game_id} className={`clickable ${selected?.game_id === g.game_id ? "selected" : ""}`} onClick={() => setSelected(g)}>
                <td>{weekLabel(g.season, g.week)}{g.season_type === "POST" ? " (playoffs)" : ""}</td>
                <td>{g.opponent ?? "—"}</td>
                <td className="num">{signed(g.rating)}</td>
                <td className="num secondary">{signed(g.matchup_rating)}</td>
                <td className="num">{g.upcoming ? "upcoming" : signed(g.actual)}</td>
                <td className="num">{g.dropbacks ?? "—"}</td>
                <td><ConfidenceBadge level={g.confidence} evidence={g.evidence} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
