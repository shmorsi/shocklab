// Typed client for the Shock Lab FastAPI service.
export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Shock = { oil_pct: number; rates_bp: number; usd_pct: number; vix_pct: number; credit_bp: number };
export type Regime = { vix_level: number; avg_corr: number };
export type ModelKey = "baseline" | "mdn";

export type Contribution = { asset: string; weight: number; cvar_contrib: number; expected_contrib: number };
export type ModelStress = {
  expected: number;
  var95: number;
  cvar95: number;
  biggest_drag: string;
  histogram: number[];
  contributions: Contribution[];
  factor_attribution: { factor: string; contribution: number }[];
  asset_factor: ({ asset: string } & Record<string, number>)[];
};
export type StressResponse = { bin_edges: number[]; models: Record<ModelKey, ModelStress>; shock: Shock; regime: Regime };

export type PortfolioRisk = { histogram: number[]; var95: number; cvar95: number; expected_stress: number; expected_base: number };
export type OptimizeResponse = {
  status: string;
  mode: string;
  model: string;
  turnover: number;
  weights: Record<string, number>;
  trades: { asset: string; from: number; to: number; delta: number }[];
  bin_edges: number[];
  before: PortfolioRisk;
  after: PortfolioRisk;
  full_optimum_cvar95: number;
  frontier: { exp_ret: number; cvar: number }[];
};

export type BandPoint = { day: number; mean: number; q05: number; q25: number; q75: number; q95: number };
export type EventModel = {
  expected: number;
  var95: number;
  cvar95: number;
  histogram: number[];
  band: BandPoint[];
  assets: { asset: string; pred: number; lo: number; hi: number; realized: number }[];
};
export type EventReplay = {
  name: string;
  start: string;
  end: string;
  shock: Shock;
  regime: Regime;
  realized_port: number;
  bin_edges: number[];
  path: { day: number; date: string; value: number }[];
  models: Record<ModelKey, EventModel>;
};

export type Meta = {
  as_of: string;
  regime_date: string;
  assets: string[];
  regime: Regime;
  mdn_trained_through: string;
  credit_series: string;
};
export type ComparisonRow = Record<string, string | number | boolean | null>;
export type Backtest = {
  metrics: Record<string, string | number>[];
  dates: string[];
  equity: Record<string, number[]>;
  last_weights: Record<string, Record<string, number>>;
};

async function req<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => req<Meta>("/meta"),
  events: () => req<EventReplay[]>("/events"),
  backtest: () => req<Backtest>("/backtest"),
  comparison: () => req<ComparisonRow[]>("/comparison"),
  stress: (weights: Record<string, number>, shock: Shock, regime: Regime | null, signal?: AbortSignal) =>
    req<StressResponse>("/stress", { weights, shock, regime }, signal),
  optimize: (body: { weights: Record<string, number>; shock: Shock; model: ModelKey; mode: string }) =>
    req<OptimizeResponse>("/optimize", body),
};
