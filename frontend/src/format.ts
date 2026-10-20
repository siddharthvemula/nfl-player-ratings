// Plain-English names for the model's six features (src/models/final.py, FINAL_FEATURES).
export const FEATURE_LABELS: Record<string, string> = {
  std_avg: "This season's EPA/dropback",
  std_dropbacks: "Dropbacks this season",
  last_avg: "Last season's EPA/dropback",
  last_dropbacks: "Dropbacks last season",
  def_shrunk: "Opponent defense",
  sack_carry: "Sack rate",
};

/** How a feature's raw value is shown (rates as %, counts as integers). */
export function featureValue(name: string, v: number): string {
  if (name === "sack_carry") return `${(v * 100).toFixed(1)}%`;
  if (name.endsWith("dropbacks")) return Math.round(v).toString();
  return signed(v);
}

/** EPA per dropback with an explicit sign, so direction never relies on color. */
export function signed(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined) return "—";
  if (Math.abs(v) < 0.5 * 10 ** -digits) v = 0; // no "−0.000"
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : v === 0 ? s : `−${s.slice(1)}`;
}

export const weekLabel = (season: number, week: number) => `${season} · Wk ${week}`;

// A typical QB-game: median dropbacks in games with 15+ dropbacks, 2019-2026 (site data).
export const TYPICAL_DROPBACKS = 36;

/**
 * A rating in plain words. The rating is expected EPA per dropback, and "average" is the league average
 * (about +0.06), not zero: a +0.24 QB is about 0.18 points per dropback better than an average one.
 */
export function ratingMeaning(rating: number, leagueAvg: number) {
  const diff = rating - leagueAvg;
  const perGame = diff * TYPICAL_DROPBACKS;
  const side = diff >= 0 ? "above" : "below";
  return {
    diff,
    perGame,
    short: `${Math.abs(diff).toFixed(2)} ${side} average · ≈ ${perGame >= 0 ? "+" : "−"}${Math.abs(perGame).toFixed(1)} pts/game`,
    long:
      `${signed(rating, 2)} = he's expected to add ${rating.toFixed(2)} points per dropback, ` +
      `${Math.abs(diff).toFixed(2)} ${diff >= 0 ? "more" : "less"} than an average QB (${signed(leagueAvg, 2)}). ` +
      `Over a typical ${TYPICAL_DROPBACKS}-dropback game that's about ${Math.abs(perGame).toFixed(1)} points ${diff >= 0 ? "more" : "fewer"}.`,
  };
}

/** The 2-letter fallback for a missing headshot. */
export const initials = (name: string) =>
  name.split(/[\s.]+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("").toUpperCase();
