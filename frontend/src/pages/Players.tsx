import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useApi, type PlayerSummary } from "../api";
import { Headshot, TeamChip } from "../components/Identity";

export function Players() {
  const { data, error } = useApi<PlayerSummary[]>("/api/players");
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const s = q.trim().toUpperCase();
    return (data ?? []).filter((p) => !s || p.name.toUpperCase().includes(s) || p.passer.toUpperCase().includes(s) || p.team.includes(s));
  }, [data, q]);
  return (
    <>
      <div className="eyebrow">Since 2019</div>
      <h1>Quarterbacks</h1>
      <p className="lede">Every QB the model has rated, most recently active first.</p>
      <div className="controls">
        <input type="search" placeholder="Search by name or team" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search quarterbacks" />
        {data && <span className="muted small">{rows.length} of {data.length}</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {!data && !error && <div className="players-grid">{Array.from({ length: 12 }, (_, i) => <div key={i} className="skeleton" style={{ height: 68 }} />)}</div>}
      <div className="players-grid">
        {rows.map((p) => (
          <Link key={p.passer_id} to={`/player/${p.passer_id}`} className="card player-card lift">
            <Headshot src={p.headshot} name={p.name} team={p.team} size={44} />
            <div style={{ minWidth: 0 }}>
              <div className="name">{p.name}</div>
              <div className="small muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <TeamChip team={p.team} />
                {p.first_season === p.last_season ? p.first_season : `${p.first_season}-${p.last_season}`} · {p.games} {p.games === 1 ? "game" : "games"}
              </div>
            </div>
          </Link>
        ))}
      </div>
      {data && !rows.length && <p className="muted">No QBs match “{q}”.</p>}
    </>
  );
}
