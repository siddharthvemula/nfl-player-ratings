import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useApi, type PlayerSummary } from "../api";

export function Players() {
  const { data, error } = useApi<PlayerSummary[]>("/api/players");
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const s = q.trim().toUpperCase();
    return (data ?? []).filter((p) => !s || p.passer.toUpperCase().includes(s) || p.team.includes(s));
  }, [data, q]);
  return (
    <>
      <h1>Quarterbacks</h1>
      <p className="secondary">Every QB the model has rated since 2019, most recently active first.</p>
      <div className="controls">
        <input type="search" placeholder="Search by name or team" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search quarterbacks" />
      </div>
      {error && <p className="error">{error}</p>}
      <div className="table-wrap">
        <table>
          <thead><tr><th>QB</th><th>Team</th><th>Seasons</th><th className="num">Games rated</th></tr></thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.passer_id}>
                <td><Link to={`/player/${p.passer_id}`}>{p.passer}</Link></td>
                <td className="secondary">{p.team}</td>
                <td className="secondary">{p.first_season === p.last_season ? p.first_season : `${p.first_season}-${p.last_season}`}</td>
                <td className="num">{p.games}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
