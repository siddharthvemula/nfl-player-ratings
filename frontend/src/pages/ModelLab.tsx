import { useState } from "react";
import { useApi, type ExperimentSection, type Meta } from "../api";

type Filter = "all" | "keep" | "drop";
const STATUS_LABEL = { keep: "✓ keep", drop: "✕ drop", reference: "• reference" } as const;
const WIDE = new Set(["change", "hypothesis", "keep/drop", "model"]);

/** Markdown-lite: **bold** and `code` are the only inline marks used in experiments.md. */
function Inline({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**") ? <strong key={i}>{p.slice(2, -2)}</strong> : p.startsWith("`") ? <code key={i}>{p.slice(1, -1)}</code> : <span key={i}>{p}</span>,
      )}
    </>
  );
}

export function ModelLab() {
  const { data, error } = useApi<ExperimentSection[]>("/api/experiments");
  const meta = useApi<Meta>("/api/meta");
  const [filter, setFilter] = useState<Filter>("all");
  const t = meta.data?.test_2025;
  const counts = (data ?? []).flatMap((s) => s.rows).reduce((c, r) => ({ ...c, [r.status]: (c[r.status] ?? 0) + 1 }), {} as Record<string, number>);

  return (
    <>
      <h1>Model Lab</h1>
      <p className="secondary">
        Every experiment, in the order it ran, straight from <code>experiments.md</code>. Each idea was kept only if it beat the
        current best in at least 80% of bootstrapped seasons <em>and</em> in at least 4 of 6 held-out seasons (2019-2024),
        each trained only on earlier seasons. {counts.keep ?? 0} kept, {counts.drop ?? 0} dropped.
      </p>

      {t && (
        <>
          <h2>Final test: 2025 (run once, model frozen beforehand)</h2>
          <div className="hero">
            <div className="card stat">
              <div className="label">League average (one number for everyone)</div>
              <div className="value">{t.weighted_mae.league_average.toFixed(4)}</div>
              <div className="sub">average miss, EPA/dropback</div>
            </div>
            <div className="card stat">
              <div className="label">Carry-over (QB's own shrunk average)</div>
              <div className="value">{t.weighted_mae.carry_over.toFixed(4)}</div>
              <div className="sub">beats league average in {(t.win_rate_vs_league_average * 100).toFixed(1)}% of resamples (final)</div>
            </div>
            <div className="card stat final">
              <div className="label">Final model</div>
              <div className="value">{t.weighted_mae.final.toFixed(4)}</div>
              <div className="sub">beats carry-over in {(t.win_rate_vs_carry_over * 100).toFixed(1)}% of resamples · gain +{t.gain_vs_carry_over.gain.toFixed(4)} (95%: +{t.gain_vs_carry_over.low.toFixed(4)} to +{t.gain_vs_carry_over.high.toFixed(4)})</div>
            </div>
          </div>
          <p className="secondary">
            {t.games} QB-games. The gain is almost all on QBs with little history: {t.low_history.games} games went from{" "}
            {t.low_history.carry_over.toFixed(4)} to {t.low_history.final.toFixed(4)}; for veterans ({t.veterans.games} games) it's a wash
            ({t.veterans.carry_over.toFixed(4)} → {t.veterans.final.toFixed(4)}).
          </p>
        </>
      )}

      <div className="controls">
        <span className="muted">Show</span>
        {(["all", "keep", "drop"] as Filter[]).map((f) => (
          <button key={f} className="chip" aria-pressed={filter === f} onClick={() => setFilter(f)}>
            {f === "all" ? "All" : f === "keep" ? "Kept" : "Dropped"}
          </button>
        ))}
      </div>
      {error && <p className="error">{error}</p>}

      {(data ?? []).map((s, i) => {
        const rows = s.rows.filter((r) => filter === "all" || r.status === filter);
        if (!rows.length) return null;
        const cols = s.columns.filter((c) => c !== "date");
        return (
          <section key={i}>
            <h2>{s.title}</h2>
            {s.notes && <p className="notes"><Inline text={s.notes} /></p>}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Call</th>
                    {cols.map((c) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, j) => (
                    <tr key={j}>
                      <td><span className={`badge ${r.status}`}>{STATUS_LABEL[r.status]}</span></td>
                      {cols.map((c) => (
                        <td key={c} className={`lab-cell ${WIDE.has(c) ? "wide" : ""}`}><Inline text={r.cells[c] ?? ""} /></td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {s.after && <p className="notes"><Inline text={s.after} /></p>}
          </section>
        );
      })}
    </>
  );
}
