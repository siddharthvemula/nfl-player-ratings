import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useApi, type Meta, type Player as PlayerT, type Rating } from "../api";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { Headshot, RankDelta, TeamChip, useTeamColor } from "../components/Identity";
import { RatingChart } from "../components/RatingChart";
import { ShapWaterfall } from "../components/ShapWaterfall";
import { ratingMeaning, signed } from "../format";

const gameLabel = (g: Rating) => `${g.season} · Week ${g.week}${g.opponent ? ` vs ${g.opponent}` : ""}`;

export function Player() {
  const { id } = useParams();
  const { data, error } = useApi<PlayerT>(id ? `/api/players/${id}` : null);
  const meta = useApi<Meta>("/api/meta");
  const teamColor = useTeamColor();
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
  if (!data) return <div className="stack"><div className="skeleton" style={{ height: 150 }} /><div className="skeleton" style={{ height: 360 }} /></div>;

  const games = season === "all" ? data.games : data.games.filter((g) => g.season === season);
  const latest = data.games[data.games.length - 1];
  const played = data.games.filter((g) => g.actual !== null);
  const features = meta.data?.features ?? Object.keys(latest.contributions);
  const avgFor = (s: number) => meta.data?.models.find((m) => m.season === s)?.league_avg ?? 0.06;
  const meaning = ratingMeaning(latest.rating, avgFor(latest.season));
  const teamName = meta.data?.teams?.[data.team]?.name;

  return (
    <>
      <p className="small"><Link to="/">← Rankings</Link></p>

      <section className="card player-hero" style={{ ["--team" as string]: teamColor(data.team) }}>
        <Headshot src={data.headshot} name={data.name} team={data.team} size={88} />
        <div>
          <div className="eyebrow">{teamName ?? data.team}</div>
          <h1>{data.name}</h1>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 14px", alignItems: "center", marginTop: 6 }}>
            <TeamChip team={data.team} />
            <ConfidenceBadge level={latest.confidence} evidence={latest.evidence} />
            {latest.upcoming && <span className="small secondary">#{latest.rank} this week <RankDelta rank={latest.rank} prev={latest.prev_rank} /></span>}
            <span className="small muted">{played.length} rated games{played.length ? ` · ${played[0].season}-${played[played.length - 1].season}` : ""}</span>
          </div>
        </div>
        <div className="rating-box">
          <div className="big tnum">{signed(latest.rating)}</div>
          <div className="big-sub">{latest.upcoming ? `rating for week ${latest.week}` : `last rating (${latest.season} wk ${latest.week})`}</div>
          {latest.opponent && <div className="big-sub">{signed(latest.matchup_rating)} vs {latest.opponent}</div>}
        </div>
      </section>

      <div className="explain" style={{ marginTop: 16 }}>
        <span className="i" aria-hidden="true">i</span>
        <span><strong>What {signed(latest.rating, 2)} means.</strong> {meaning.long}</span>
      </div>

      <section className="card section">
        <div className="card-head">
          <div>
            <h2 style={{ margin: 0 }}>Rating over time</h2>
            <div className="muted small">Each point is the forecast made before that game; dots are what happened. Click a game.</div>
          </div>
          <select value={season} onChange={(e) => setSeason(e.target.value === "all" ? "all" : Number(e.target.value))} aria-label="Season">
            {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
            <option value="all">All seasons</option>
          </select>
        </div>
        <RatingChart games={games} selected={selected} onSelect={setSelected} leagueAvg={avgFor(typeof season === "number" ? season : latest.season)} />
      </section>

      {selected && (
        <section className="card section" id="why" style={{ scrollMarginTop: 80 }}>
          <div className="card-head">
            <div>
              <h2 style={{ margin: 0 }}>Why {signed(selected.matchup_rating)}?</h2>
              <div className="muted small">{gameLabel(selected)}{selected.actual !== null ? ` · actual ${signed(selected.actual)}` : " · not played yet"}</div>
            </div>
          </div>
          <p className="secondary small" style={{ maxWidth: "80ch" }}>
            Every rating starts from <strong>carry-over</strong>: his EPA/dropback, last season blended into this one and pulled toward the
            league average when there's little to go on. Each bar after that is one feature's push (weight × how far his value is from a
            typical game). The pieces add up exactly to the rating.
          </p>
          <ShapWaterfall game={selected} order={features} />
          <div className="legend" style={{ marginTop: 14, marginBottom: 0 }}>
            <span><span className="sq" style={{ background: "var(--up)" }} />raises the rating (+)</span>
            <span><span className="sq" style={{ background: "var(--down)" }} />lowers it (−)</span>
            <span><span className="sq" style={{ background: "var(--total)" }} />starting point / result</span>
          </div>
        </section>
      )}

      <h2>Games</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Game</th><th className="num">Rating</th><th className="num hide-sm">vs opp</th>
              <th className="num">Actual</th><th className="num hide-sm">Dropbacks</th><th className="hide-sm">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {[...games].reverse().map((g) => (
              <tr key={g.game_id} className={`clickable ${selected?.game_id === g.game_id ? "selected" : ""}`}
                onClick={() => { setSelected(g); document.getElementById("why")?.scrollIntoView({ behavior: "smooth" }); }}>
                <td>{gameLabel(g)}{g.season_type === "POST" ? <span className="badge" style={{ marginLeft: 6 }}>playoffs</span> : null}</td>
                <td className="num">{signed(g.rating)}</td>
                <td className="num secondary hide-sm">{signed(g.matchup_rating)}</td>
                <td className="num">{g.upcoming ? <span className="muted">upcoming</span> : signed(g.actual)}</td>
                <td className="num hide-sm">{g.dropbacks ?? "—"}</td>
                <td className="hide-sm"><ConfidenceBadge level={g.confidence} evidence={g.evidence} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
