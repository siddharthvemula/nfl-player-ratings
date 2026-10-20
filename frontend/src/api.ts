// Types mirror the FastAPI response shapes in api/index.py.
import { useEffect, useState } from "react";

export type Confidence = "Low" | "Med" | "High";

export interface Rating {
  season: number;
  week: number;
  season_type: string;
  game_id: string;
  passer_id: string;
  passer: string; // short name, as in the play-by-play ("J.Herbert")
  name: string; // full name ("Justin Herbert")
  headshot: string | null;
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
  raw_epa: number | null; // raw EPA/dropback this season before this game (null before his first dropback)
  season_dropbacks: number;
  prev_rank: number | null; // his most recent earlier rank this season
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
  name: string;
  headshot: string | null;
  team: string;
  first_season: number;
  last_season: number;
  games: number;
}

export interface Player {
  passer_id: string;
  passer: string;
  name: string;
  headshot: string | null;
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
    calibration: { bin: number; games: number; predicted: number; actual: number; se: number }[];
  };
  teams: Record<string, { name: string; color: string; color2: string }>;
}

export interface TestGame {
  passer_id: string;
  name: string;
  team: string;
  week: number;
  opponent: string | null;
  predicted: number;
  actual: number;
  dropbacks: number;
}

export interface ExperimentSection {
  title: string;
  level: number;
  notes: string;
  after: string;
  columns: string[];
  rows: { cells: Record<string, string>; status: "keep" | "drop" | "reference" }[];
}

// One request per path per page load: pages share meta and revisit weeks without re-fetching.
const cache = new Map<string, Promise<unknown>>();

export function fetchJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    const p = fetch(path).then(async (r) => {
      if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? `HTTP ${r.status}`);
      return r.json();
    });
    p.catch(() => cache.delete(path)); // let a failed request be retried
    cache.set(path, p);
  }
  return cache.get(path) as Promise<T>;
}

type State<T> = { data: T | null; error: string | null; loading: boolean };

/** Fetch JSON from the API; re-fetches when the path changes. */
export function useApi<T>(path: string | null): State<T> {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fetchJson<T>(path)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e) => alive && setState({ data: null, error: String(e.message ?? e), loading: false }));
    return () => {
      alive = false;
    };
  }, [path]);
  return state;
}

/** Several paths at once (e.g. every week of a season for the slider); data arrives when all have loaded. */
export function useApiMany<T>(paths: string[]): State<T[]> {
  const key = paths.join("|");
  const [state, setState] = useState<State<T[]>>({ data: null, error: null, loading: true });
  useEffect(() => {
    if (!paths.length) return;
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    Promise.all(paths.map((p) => fetchJson<T>(p)))
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e) => alive && setState({ data: null, error: String(e.message ?? e), loading: false }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
