import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useApi, type Confidence, type Meta, type Rankings as RankingsT } from "../api";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { signed, weekLabel } from "../format";

type SortKey = "rank" | "matchup_rating" | "evidence";

export function Rankings() {
  const meta = useApi<Meta>("/api/meta");
  const [params, setParams] = useSearchParams();
  const season = params.get("season");
  const week = params.get("week");
  const path = season && week ? `/api/rankings?season=${season}&week=${week}` : "/api/rankings";
  const { data, error, loading } = useApi<RankingsT>(path);
  const [conf, setConf] = useState<Set<Confidence>>(new Set(["Low", "Med", "High"]));
  const [team, setTeam] = useState("");
  const [sort, setSort] = useState<SortKey>("rank");

  // Week picker: the current upcoming week first, then every played week, newest first.
  const weeks = useMemo(() => {
    if (!meta.data) return [];
    const played = [...meta.data.played_weeks].reverse().map((w) => ({ ...w, upcoming: false }));
    return meta.data.current.upcoming ? [{ ...meta.data.current }, ...played] : played;
  }, [meta.data]);

  const rows = useMemo(() => {
    if (!data) return [];
    const q = team.trim().toUpperCase();
    const out = data.rows.filter((r) => conf.has(r.confidence) && (!q || r.team.includes(q) || r.passer.toUpperCase().includes(q)));
    if (sort === "matchup_rating") out.sort((a, b) => b.matchup_rating - a.matchup_rating);
    if (sort === "evidence") out.sort((a, b) => b.evidence - a.evidence);
    return out;
  }, [data, conf, team, sort]);

  const span = useMemo(() => {
    if (!data?.rows.length) return { lo: -0.2, hi: 0.3 };
    const vs = data.rows.map((r) => r.rating);
    return { lo: Math.min(0, ...vs), hi: Math.max(0, ...vs) };
  }, [data]);
  const pct = (v: number) => ((v - span.lo) / (span.hi - span.lo || 1)) * 100;

  const toggle = (c: Confidence) => setConf((s) => {
    const n = new Set(s);
    if (n.has(c)) n.delete(c); else n.add(c);
    return n;
  });

  return (
    <>
      <h1>QB rankings</h1>
      <p className="secondary">
        Each rating is the model's forecast of a QB's EPA per dropback in his next game, made before it's played.
        Rankings use the <strong>opponent-neutral</strong> rating (an average defense), so the order is about the QB, not the schedule.
        {" "}<Link to="/method">How it works</Link>
      </p>

      <div className="controls">
        <label>
          <span className="muted">Week </span>
          <select
            value={data ? `${data.season}-${data.week}` : ""}
            onChange={(e) => {
              const [s, w] = e.target.value.split("-");
              const w0 = weeks[0];
              if (w0 && Number(s) === w0.season && Number(w) === w0.week && w0.upcoming) setParams({});
              else setParams({ season: s, week: w });
            }}
          >
            {weeks.map((w) => (
              <option key={`${w.season}-${w.week}`} value={`${w.season}-${w.week}`}>
                {weekLabel(w.season, w.week)}{w.upcoming ? " (upcoming)" : ""}
              </option>
            ))}
          </select>
        </label>
        <input type="search" placeholder="Filter by team or QB" value={team} onChange={(e) => setTeam(e.target.value)} aria-label="Filter by team or QB" />
        <span className="muted">Confidence</span>
        {(["High", "Med", "Low"] as Confidence[]).map((c) => (
          <button key={c} className="chip" aria-pressed={conf.has(c)} onClick={() => toggle(c)}>{c}</button>
        ))}
      </div>

      {error && <p className="error">{error}</p>}
      {loading && !data && <p className="muted">Loading…</p>}
      {data && (
        <>
          <p className="muted">
            {data.upcoming
              ? `Ratings for ${weekLabel(data.season, data.week)}, made from games through the previous week. QBs on a bye show their next game.`
              : `Ratings made before ${weekLabel(data.season, data.week)}, next to what each QB actually did.`}
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th><button onClick={() => setSort("rank")}>#{sort === "rank" ? " ↓" : ""}</button></th>
                  <th>QB</th>
                  <th>Next game</th>
                  <th className="num"><button onClick={() => setSort("rank")}>Rating</button></th>
                  <th className="num"><button onClick={() => setSort("matchup_rating")} title="Forecast against the actual opponent">vs opponent{sort === "matchup_rating" ? " ↓" : ""}</button></th>
                  {!data.upcoming && <th className="num">Actual</th>}
                  <th><button onClick={() => setSort("evidence")} title="Dropbacks of evidence: last season + this season so far">Confidence{sort === "evidence" ? " ↓" : ""}</button></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.passer_id + r.game_id}>
                    <td className="num">{r.rank}</td>
                    <td>
                      <Link to={`/player/${r.passer_id}`}>{r.passer}</Link> <span className="muted">{r.team}</span>
                    </td>
                    <td className="secondary">
                      {r.opponent ? `${r.opponent}` : "—"}
                      {data.upcoming && r.week !== data.week ? <span className="muted"> (Wk {r.week}, bye)</span> : null}
                    </td>
                    <td className="num">
                      {signed(r.rating)}
                      <div className="ratingbar" aria-hidden="true">
                        <span className="zero" style={{ left: `${pct(0)}%` }} />
                        <span style={{ left: `${Math.min(pct(0), pct(r.rating))}%`, width: `${Math.abs(pct(r.rating) - pct(0))}%` }} />
                      </div>
                    </td>
                    <td className="num secondary">{signed(r.matchup_rating)}</td>
                    {!data.upcoming && <td className="num">{signed(r.actual)}</td>}
                    <td><ConfidenceBadge level={r.confidence} evidence={r.evidence} /></td>
                  </tr>
                ))}
                {!rows.length && (
                  <tr><td colSpan={7} className="muted">No QBs match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {meta.data && (
        <>
          <h2>What confidence means</h2>
          <p className="secondary">
            Confidence is how much evidence the model has: dropbacks last season plus this season so far. Rookies and backups start out Low.
            Average miss of the ratings (EPA/dropback, weighted by dropbacks) in past seasons, 2019-2024:
          </p>
          <div className="cards">
            {meta.data.confidence_tiers.map((t) => (
              <div className="card stat" key={t.tier}>
                <div className="label"><ConfidenceBadge level={t.tier} /> {t.max_evidence ? `${t.min_evidence}-${t.max_evidence}` : `${t.min_evidence}+`} dropbacks</div>
                <div className="value">{t.weighted_mae.toFixed(3)}</div>
                <div className="sub">average miss · {t.games.toLocaleString()} games</div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}
