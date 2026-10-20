import { useCallback, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useApi, useApiMany, type Confidence, type Meta, type Rankings as RankingsT, type Rating } from "../api";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { Headshot, RankDelta, TeamChip, useTeamColor } from "../components/Identity";
import { QbLandscape } from "../components/QbLandscape";
import { SortHeader, type SortState } from "../components/SortHeader";
import { ratingMeaning, signed } from "../format";

type SortKey = "rank" | "rating" | "delta" | "name" | "team" | "matchup_rating" | "raw_epa" | "actual" | "evidence";

// Each column's sort value; nulls always sink to the bottom.
const SORT_VALUE: Record<SortKey, (r: Rating) => number | string | null> = {
  rank: (r) => r.rank,
  rating: (r) => r.rating,
  delta: (r) => (r.prev_rank === null ? null : r.prev_rank - r.rank),
  name: (r) => r.name,
  team: (r) => r.team,
  matchup_rating: (r) => r.matchup_rating,
  raw_epa: (r) => r.raw_epa,
  actual: (r) => r.actual,
  evidence: (r) => r.evidence,
};

export function Rankings() {
  const meta = useApi<Meta>("/api/meta");
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const teamColor = useTeamColor();

  // Seasons with ratings, newest first; the live season by default.
  const seasons = useMemo(() => [...new Set(meta.data?.played_weeks.map((w) => w.season) ?? [])].reverse(), [meta.data]);
  const season = Number(params.get("season")) || meta.data?.current.season || 0;

  // Every ranking of the season, in order: played weeks, then the upcoming week (live season only).
  const paths = useMemo(() => {
    if (!meta.data) return [];
    const played = meta.data.played_weeks.filter((w) => w.season === season).map((w) => `/api/rankings?season=${w.season}&week=${w.week}`);
    const cur = meta.data.current;
    return cur.upcoming && cur.season === season ? [...played, "/api/rankings"] : played;
  }, [meta.data, season]);
  const weeks = useApiMany<RankingsT>(paths);

  const n = weeks.data?.length ?? 0;
  const wantWeek = Number(params.get("week"));
  const found = weeks.data?.findIndex((w) => w.week === wantWeek) ?? -1;
  const index = found >= 0 ? found : Math.max(0, n - 1);
  const setIndex = useCallback((i: number) => {
    const w = weeks.data?.[i];
    if (!w) return;
    setParams((p) => {
      const q = new URLSearchParams(p);
      q.set("season", String(w.season));
      q.set("week", String(w.week));
      return q;
    }, { replace: true });
  }, [weeks.data, setParams]);
  const data = weeks.data?.[index];

  const [conf, setConf] = useState<Set<Confidence>>(new Set(["Low", "Med", "High"]));
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortState<SortKey>>({ key: "rank", dir: "asc" });

  const rows = useMemo(() => {
    if (!data) return [];
    const s = q.trim().toUpperCase();
    const out = data.rows.filter((r) => conf.has(r.confidence) && (!s || r.team.includes(s) || r.name.toUpperCase().includes(s)));
    const val = SORT_VALUE[sort.key];
    const sign = sort.dir === "asc" ? 1 : -1;
    return out.sort((a, b) => {
      const [x, y] = [val(a), val(b)];
      if (x === null || y === null) return x === null ? (y === null ? a.rank - b.rank : 1) : -1;
      return (x < y ? -1 : x > y ? 1 : a.rank - b.rank) * sign;
    });
  }, [data, conf, q, sort]);

  const span = useMemo(() => {
    const vs = data?.rows.map((r) => r.rating) ?? [0];
    return { lo: Math.min(0, ...vs), hi: Math.max(0, ...vs) };
  }, [data]);
  const pct = (v: number) => ((v - span.lo) / (span.hi - span.lo || 1)) * 100;

  const model = meta.data?.models.find((m) => m.season === season) ?? meta.data?.models[meta.data.models.length - 1];
  const leagueAvg = model?.league_avg ?? 0.06;
  const top3 = data ? [...data.rows].sort((a, b) => a.rank - b.rank).slice(0, 3) : [];
  const toggle = (c: Confidence) => setConf((s) => {
    const next = new Set(s);
    if (next.has(c)) next.delete(c); else next.add(c);
    return next;
  });

  return (
    <>
      <div className="eyebrow">{data ? `${data.season} · Week ${data.week}${data.upcoming ? " · upcoming" : ""}` : "Weekly ratings"}</div>
      <h1>QB rankings</h1>
      <p className="lede">
        Each rating is the model's forecast of a QB's EPA per dropback in his next game, made before it's played. Rankings use
        the <strong>opponent-neutral</strong> rating (an average defense), so the order is about the QB, not the schedule.{" "}
        <Link to="/method">How it works</Link>
      </p>

      {(meta.error || weeks.error) && <p className="error">{meta.error ?? weeks.error}</p>}
      {!data && !weeks.error && <Loading />}

      {data && (
        <>
          <div className="podium">
            {top3.map((r) => {
              const m = ratingMeaning(r.rating, leagueAvg);
              return (
                <Link key={r.passer_id} to={`/player/${r.passer_id}`} className="card lift" style={{ ["--team" as string]: teamColor(r.team) }}>
                  <div className="top">
                    <Headshot src={r.headshot} name={r.name} team={r.team} size={48} />
                    <div>
                      <div className="rank">#{r.rank}</div>
                      <div className="name">{r.name}</div>
                    </div>
                    <span style={{ marginLeft: "auto" }}><TeamChip team={r.team} /></span>
                  </div>
                  <div className="rating tnum">{signed(r.rating)}<small>EPA/dropback</small></div>
                  <div className="meaning">{m.short}</div>
                  <div className="foot">
                    <ConfidenceBadge level={r.confidence} evidence={r.evidence} />
                    <span className="small muted">vs last <RankDelta rank={r.rank} prev={r.prev_rank} /></span>
                  </div>
                </Link>
              );
            })}
          </div>

          {top3[0] && (
            <div className="explain">
              <span className="i" aria-hidden="true">i</span>
              <span><strong>Reading a rating.</strong> {ratingMeaning(top3[0].rating, leagueAvg).long}</span>
            </div>
          )}

          <section className="card section">
            <div className="card-head">
              <div>
                <h2 style={{ margin: 0 }}>QB landscape</h2>
                <div className="muted small">Raw production this season vs what the model expects next</div>
              </div>
              <select value={season} aria-label="Season" onChange={(e) => setParams({ season: e.target.value })}>
                {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            {weeks.data && <QbLandscape weeks={weeks.data} index={index} onIndex={setIndex} leagueAvg={leagueAvg} seasonLabel={String(season)} />}
          </section>

          <h2>{data.upcoming ? `Week ${data.week} rankings` : `Before week ${data.week}`}</h2>
          <p className="muted small">
            {data.upcoming
              ? "Made from games through the previous week. QBs on a bye show their next game. Arrows compare with his previous ranking this season."
              : "Ratings made before these games, next to what each QB actually did. Arrows compare with his previous ranking this season."}
          </p>
          <div className="controls">
            <input type="search" placeholder="Search QB or team" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search QB or team" />
            <span className="muted small">Confidence</span>
            <span className="segmented">
              {(["High", "Med", "Low"] as Confidence[]).map((c) => (
                <button key={c} aria-pressed={conf.has(c)} onClick={() => toggle(c)}>{c}</button>
              ))}
            </span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <SortHeader label="#" k="rank" sort={sort} onSort={setSort} natural="asc" />
                  <SortHeader label="Δ" k="delta" sort={sort} onSort={setSort} title="Rank change vs his previous ranking" />
                  <SortHeader label="QB" k="name" sort={sort} onSort={setSort} natural="asc" />
                  <SortHeader label={data.upcoming ? "Next" : "Opp"} k="team" sort={sort} onSort={setSort} natural="asc" className="hide-sm" />
                  <SortHeader label="Rating" k="rating" sort={sort} onSort={setSort} num />
                  <SortHeader label="vs opp" k="matchup_rating" sort={sort} onSort={setSort} num className="hide-sm" title="Forecast against the actual opponent" />
                  <SortHeader label="Raw EPA" k="raw_epa" sort={sort} onSort={setSort} num className="hide-sm" title="His raw EPA/dropback this season so far" />
                  {!data.upcoming && <SortHeader label="Actual" k="actual" sort={sort} onSort={setSort} num />}
                  <SortHeader label="Confidence" k="evidence" sort={sort} onSort={setSort} title="Dropbacks of evidence: last season + this season so far" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.passer_id + r.game_id} className="clickable" onClick={() => navigate(`/player/${r.passer_id}`)}>
                    <td className="num" style={{ fontWeight: 700 }}>{r.rank}</td>
                    <td><RankDelta rank={r.rank} prev={r.prev_rank} /></td>
                    <td>
                      <div className="qb-cell">
                        <span className="hide-sm"><Headshot src={r.headshot} name={r.name} team={r.team} size={32} /></span>
                        <div className="who">
                          <Link to={`/player/${r.passer_id}`} onClick={(e) => e.stopPropagation()}>{r.name}</Link>{" "}
                          <TeamChip team={r.team} />
                        </div>
                      </div>
                    </td>
                    <td className="secondary hide-sm">
                      {r.opponent ? r.opponent : "—"}
                      {data.upcoming && r.week !== data.week ? <span className="muted"> (Wk {r.week}, bye)</span> : null}
                    </td>
                    <td className="num" style={{ fontWeight: 600 }}>
                      {signed(r.rating)}
                      <div className="ratingbar" aria-hidden="true">
                        <span className="zero" style={{ left: `${pct(0)}%` }} />
                        <span style={{ left: `${Math.min(pct(0), pct(r.rating))}%`, width: `${Math.abs(pct(r.rating) - pct(0))}%` }} />
                      </div>
                    </td>
                    <td className="num secondary hide-sm">{signed(r.matchup_rating)}</td>
                    <td className="num secondary hide-sm">{signed(r.raw_epa)}</td>
                    {!data.upcoming && <td className="num">{signed(r.actual)}</td>}
                    <td><ConfidenceBadge level={r.confidence} evidence={r.evidence} /></td>
                  </tr>
                ))}
                {!rows.length && <tr><td colSpan={9} className="muted">No QBs match these filters.</td></tr>}
              </tbody>
            </table>
          </div>
        </>
      )}

      {meta.data && (
        <>
          <h2>What confidence means</h2>
          <p className="secondary" style={{ maxWidth: "75ch" }}>
            Confidence is how much evidence the model has: dropbacks last season plus this season so far. Rookies and backups start
            out Low. Below is each tier's real average miss (EPA/dropback, weighted by dropbacks) on past seasons, 2019-2024.
          </p>
          <div className="grid cols-3">
            {meta.data.confidence_tiers.map((t) => (
              <div className="card tight stat" key={t.tier}>
                <div className="label"><ConfidenceBadge level={t.tier} /> <span className="muted">· {t.max_evidence ? `${t.min_evidence}-${t.max_evidence}` : `${t.min_evidence}+`} dropbacks</span></div>
                <div className="value tnum">{t.weighted_mae.toFixed(3)}</div>
                <div className="sub">average miss · {t.games.toLocaleString()} games</div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function Loading() {
  return (
    <div className="stack" aria-label="Loading">
      <div className="podium" style={{ margin: 0 }}>{[0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 180 }} />)}</div>
      <div className="skeleton" style={{ height: 420 }} />
    </div>
  );
}
