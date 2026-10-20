import { useState } from "react";
import { useApi, type Meta } from "../api";
import { initials } from "../format";
import { markColor, useTheme } from "../theme";

/** Team colors from nflverse (meta.teams), adjusted for the current theme's surface. */
export function useTeamColor() {
  const meta = useApi<Meta>("/api/meta");
  const { theme } = useTheme();
  return (team: string) => markColor(meta.data?.teams?.[team]?.color, theme);
}

/** Team abbreviation with a small swatch of its color. The text carries identity; the swatch is extra. */
export function TeamChip({ team }: { team: string }) {
  const color = useTeamColor();
  const meta = useApi<Meta>("/api/meta");
  return (
    <span className="team-chip" title={meta.data?.teams?.[team]?.name ?? team}>
      <i style={{ background: color(team) }} aria-hidden="true" />
      {team}
    </span>
  );
}

/** Player photo (nflverse headshot URL), ringed in his team color; initials if there's no photo or it fails. */
export function Headshot({ src, name, team, size = 40 }: { src: string | null; name: string; team?: string; size?: number }) {
  const color = useTeamColor();
  const [broken, setBroken] = useState(false);
  return (
    <span className="headshot" style={{ width: size, height: size, ["--team" as string]: team ? color(team) : undefined, fontSize: size * 0.32 }}>
      {src && !broken ? (
        <img src={src} alt="" loading="lazy" referrerPolicy="no-referrer" onError={() => setBroken(true)} />
      ) : (
        <span aria-hidden="true">{initials(name)}</span>
      )}
    </span>
  );
}

/** Rank change vs his previous rank this season: ▲ up, ▼ down, — same, NEW for his first ranking. */
export function RankDelta({ rank, prev }: { rank: number; prev: number | null }) {
  if (prev === null) return <span className="delta new" title="First ranking this season">NEW</span>;
  const d = prev - rank;
  if (d === 0) return <span className="delta same" title="Same rank as last time">—</span>;
  return (
    <span className={`delta ${d > 0 ? "up" : "down"}`} title={`${d > 0 ? "Up" : "Down"} ${Math.abs(d)} (was #${prev})`}>
      {d > 0 ? "▲" : "▼"}{Math.abs(d)}
    </span>
  );
}
