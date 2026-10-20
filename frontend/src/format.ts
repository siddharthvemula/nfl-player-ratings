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
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : v === 0 ? s : `−${s.slice(1)}`;
}

export const weekLabel = (season: number, week: number) => `${season} · Wk ${week}`;
