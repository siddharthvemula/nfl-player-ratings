// Types mirror the FastAPI response shapes in api/index.py.
import { useEffect, useState } from "react";

export type Confidence = "Low" | "Med" | "High";

export interface Rating {
  season: number;
  week: number;
  season_type: string;
  game_id: string;
  passer_id: string;
  passer: string;
  team: string;
  opponent: string | null;
  upcoming: boolean;
  rank: number;
  rating: number; // opponent-neutral (what the rankings use)
  matchup_rating: number; // vs the actual opponent
  actual: number | null;
  dropbacks: number | null;
  evidence: number;
  confidence: Confidence;
  carry_over: number;
  avg_adjustment: number;
  contributions: Record<string, number>;
  features: Record<string, number>;
}

export interface Rankings {
  season: number;
  week: number;
  upcoming: boolean;
  rows: Rating[];
}

export interface PlayerSummary {
  passer_id: string;
  passer: string;
  team: string;
  first_season: number;
  last_season: number;
  games: number;
}

export interface Player {
  passer_id: string;
  passer: string;
  team: string;
  games: Rating[];
}

export interface ModelFit {
  season: number;
  trained_on: string;
  league_avg: number;
  carry_k: number;
  defense_k: number;
  sack_k: number;
  intercept: number;
  weights: Record<string, number>;
  feature_means: Record<string, number>;
}

export interface Meta {
  updated_at: string;
  live_season: number;
  current: { season: number; week: number; upcoming: boolean };
  played_weeks: { season: number; week: number }[];
  features: string[];
  models: ModelFit[];
  confidence_tiers: { tier: Confidence; min_evidence: number; max_evidence: number | null; games: number; weighted_mae: number }[];
  test_2025: {
    games: number;
    weighted_mae: { league_average: number; carry_over: number; final: number };
    unweighted_mae: { league_average: number; carry_over: number; final: number };
    win_rate_vs_carry_over: number;
    win_rate_vs_league_average: number;
    gain_vs_carry_over: { gain: number; low: number; high: number };
    low_history: { games: number; carry_over: number; final: number };
    veterans: { games: number; carry_over: number; final: number };
  };
}

export interface ExperimentSection {
  title: string;
  level: number;
  notes: string;
  after: string;
  columns: string[];
  rows: { cells: Record<string, string>; status: "keep" | "drop" | "reference" }[];
}

/** Fetch JSON from the API; re-fetches when the path changes. */
export function useApi<T>(path: string | null): { data: T | null; error: string | null; loading: boolean } {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: true });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(path)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e) => alive && setState({ data: null, error: String(e.message ?? e), loading: false }));
    return () => {
      alive = false;
    };
  }, [path]);
  return state;
}
