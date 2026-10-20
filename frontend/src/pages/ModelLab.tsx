import { Fragment, useState } from "react";
import { useApi, type ExperimentSection, type Meta } from "../api";

type Filter = "all" | "keep" | "drop";
type Row = ExperimentSection["rows"][number];
const STATUS_LABEL = { keep: "✓ Kept", drop: "✕ Dropped", reference: "• Reference" } as const;

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

/** The most precise number in a cell: "0.233 (0.2327)" -> 0.2327, "**0.2260**" -> 0.226. */
function mae(cell: string | undefined): number | null {
  if (!cell) return null;
  const paren = cell.match(/\((\d*\.\d+)\)/);
  const first = cell.match(/\d*\.\d+/);
  const m = paren ?? first;
  return m ? Number(m[1] ?? m[0]) : null;
}

/** Which column holds a table's score, and the MAE change of each row against what it was compared with. */
function scoreColumn(s: ExperimentSection) {
  return s.columns.find((c) => c === "pooled MAE") ?? s.columns.find((c) => /MAE \(weighted\)/.test(c)) ?? null;
}
function changes(s: ExperimentSection): ({ delta: number; vs: string } | null)[] {
  const col = scoreColumn(s);
  if (!col) return s.rows.map(() => null);
  let lastKept: { mae: number; name: string } | null = null;
  return s.rows.map((r, i) => {
    const own = mae(r.cells[col]);
    let out: { delta: number; vs: string } | null = null;
    if (own !== null && "vs MAE" in r.cells) {
      // Walk-forward tables name what each row was compared with.
      const base = mae(r.cells["vs MAE"]);
      if (base !== null && r.cells.vs && r.cells.vs !== "—") out = { delta: own - base, vs: r.cells.vs };
    } else if (own !== null && lastKept && i > 0) {
      // The first (2024-only) table has no "vs" column: compare with the last kept row above it.
      out = { delta: own - lastKept.mae, vs: lastKept.name };
    }
    if (r.status === "keep" && own !== null) lastKept = { mae: own, name: shortName(r) };
    return out;
  });
}
const shortName = (r: Row) => (r.cells.change ?? r.cells.model ?? "").split(/[:(]/)[0].replace(/\*/g, "").trim().slice(0, 28);

function DeltaPill({ c }: { c: { delta: number; vs: string } | null }) {
  if (!c) return null;
  const d = Math.round(c.delta * 10000) / 10000;
  const better = d < 0;
  const cls = d === 0 ? "" : better ? "better" : "worse";
  const word = d === 0 ? "tie" : better ? "better" : "worse";
  return (
    <span className={`pill ${cls}`} title={`MAE ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(4)} vs ${c.vs} (lower is better)`}>
      {d === 0 ? "=" : better ? "▼" : "▲"} {Math.abs(d).toFixed(4)} {word} vs {c.vs}
    </span>
  );
}

function ExperimentCard({ row, change, score }: { row: Row; change: { delta: number; vs: string } | null; score: string | null }) {
  const title = row.cells.change ?? row.cells.model ?? "";
  const verdict = row.cells["keep/drop"];
  const shown = new Set(["date", "change", "model", "hypothesis", "keep/drop"]);
  const rest = Object.entries(row.cells).filter(([k, v]) => !shown.has(k) && v);
  return (
    <article className={`card tight exp lift ${row.status}`}>
      <div className="top">
        <span className={`badge ${row.status}`}>{STATUS_LABEL[row.status]}</span>
        <DeltaPill c={change} />
      </div>
      <div className="title"><Inline text={title} /></div>
      {row.cells.hypothesis && <div className="hyp"><Inline text={row.cells.hypothesis} /></div>}
      <div className="metrics">
        {score && row.cells[score] && <span>MAE <b><Inline text={row.cells[score]} /></b></span>}
        {/\d/.test(row.cells["win %"] ?? "") && <span>win <b>{row.cells["win %"]}%</b></span>}
        {/\d/.test(row.cells["seasons won"] ?? "") && <span>seasons won <b>{row.cells["seasons won"]}</b></span>}
      </div>
      {(verdict || rest.length > 0) && (
        <details>
          <summary>Details</summary>
          <dl>
            {verdict && <><dt>verdict</dt><dd><Inline text={verdict} /></dd></>}
            {rest.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd><Inline text={v} /></dd></Fragment>)}
          </dl>
        </details>
      )}
    </article>
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
      <div className="eyebrow">Every experiment, kept or not</div>
      <h1>Model Lab</h1>
      <p className="lede">
        Every experiment, in the order it ran, straight from <code>experiments.md</code>. Each idea was kept only if it beat the current best
        in at least 80% of bootstrapped seasons <em>and</em> in at least 4 of 6 held-out seasons (2019-2024), each trained only on earlier
        seasons. {counts.keep ?? 0} kept, {counts.drop ?? 0} dropped.
      </p>

      {t && (
        <>
          <h2>Final test: 2025 <span className="muted" style={{ fontWeight: 500 }}>· run once, model frozen beforehand</span></h2>
          <div className="grid cols-3">
            <div className="card stat">
              <div className="label">League average <span className="muted">(one number for everyone)</span></div>
              <div className="value tnum">{t.weighted_mae.league_average.toFixed(4)}</div>
              <div className="sub">average miss, EPA/dropback</div>
            </div>
            <div className="card stat">
              <div className="label">Carry-over <span className="muted">(QB's own shrunk average)</span></div>
              <div className="value tnum">{t.weighted_mae.carry_over.toFixed(4)}</div>
              <div className="sub">the simple baseline to beat</div>
            </div>
            <div className="card stat accent">
              <div className="label">Final model</div>
              <div className="value tnum">{t.weighted_mae.final.toFixed(4)}</div>
              <div className="sub">beats carry-over in {(t.win_rate_vs_carry_over * 100).toFixed(1)}% of resamples · gain +{t.gain_vs_carry_over.gain.toFixed(4)} (95%: +{t.gain_vs_carry_over.low.toFixed(4)} to +{t.gain_vs_carry_over.high.toFixed(4)})</div>
            </div>
          </div>
          <p className="secondary small" style={{ marginTop: 12, maxWidth: "85ch" }}>
            {t.games} QB-games. The gain is almost all on QBs with little history: {t.low_history.games} games went from{" "}
            {t.low_history.carry_over.toFixed(4)} to {t.low_history.final.toFixed(4)}. For veterans ({t.veterans.games} games) it's a wash
            ({t.veterans.carry_over.toFixed(4)} → {t.veterans.final.toFixed(4)}).
          </p>
        </>
      )}

      <h2>Experiments</h2>
      <div className="controls">
        <span className="segmented">
          {(["all", "keep", "drop"] as Filter[]).map((f) => (
            <button key={f} aria-pressed={filter === f} onClick={() => setFilter(f)}>
              {f === "all" ? "All" : f === "keep" ? "Kept" : "Dropped"}
            </button>
          ))}
        </span>
        <span className="muted small">▼ = lower error than what it was compared with (better)</span>
      </div>
      {error && <p className="error">{error}</p>}

      {(data ?? []).map((s, i) => {
        const deltas = changes(s);
        const score = scoreColumn(s);
        const rows = s.rows.map((r, j) => ({ r, c: deltas[j] })).filter(({ r }) => filter === "all" || r.status === filter);
        if (!rows.length) return null;
        return (
          <section key={i} className="section">
            <h3 style={{ fontSize: "1.05rem" }}>{s.title}</h3>
            {s.notes && <details className="notes" style={{ marginBottom: 12 }}><summary className="muted">How to read this table</summary><p style={{ marginTop: 8 }}><Inline text={s.notes} /></p></details>}
            <div className="exp-grid">
              {rows.map(({ r, c }, j) => <ExperimentCard key={j} row={r} change={c} score={score} />)}
            </div>
            {s.after && <p className="notes" style={{ marginTop: 12 }}><Inline text={s.after} /></p>}
          </section>
        );
      })}
    </>
  );
}
