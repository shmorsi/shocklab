// Browser port of shocklab/stress.py, used by the static (GitHub Pages) build.
// Same models and maths as the Python API: the baseline (alpha + beta·shock +
// multivariate-t residuals), the MDN forward pass, VaR/CVaR/attribution, and the
// Rockafellar–Uryasev LP solved with HiGHS compiled to WebAssembly.
// Random draws use a seeded JS generator, so numbers match the API statistically,
// not digit-for-digit.
import type { ModelKey, ModelStress, OptimizeResponse, Regime, Shock, StressResponse } from "./api";

const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const CONF = 0.95;
const N_STRESS = 10_000;
const N_OPT = 1_500; // fewer scenarios than the API so the LP stays fast in the browser
const N_BINS = 60;
const SEED = 7;
const HUMAN: Record<string, keyof Shock> = { oil: "oil_pct", rates: "rates_bp", usd: "usd_pct", vix: "vix_pct", credit: "credit_bp" };

type EngineData = {
  factors: string[];
  log_factors: string[];
  baseline: { assets: string[]; alpha: number[]; beta: number[][]; chol: number[][]; nu: number };
  mdn: {
    assets: string[]; k: number; log_sig_min: number; log_sig_max: number;
    x_mean: number[]; x_std: number[]; y_mean: number[]; y_std: number[];
    w1: number[][]; b1: number[]; w2: number[][]; b2: number[]; w3: number[][]; b3: number[];
  };
  regime: number[];
  shock_hist: number[][];
};

let dataPromise: Promise<EngineData> | null = null;
const loadData = (): Promise<EngineData> => (dataPromise ??= fetch(`${BASE}/data/engine.json`).then((r) => r.json() as Promise<EngineData>));

// ---------- seeded random numbers ----------
function rng(seed: number) {
  let s = seed >>> 0;
  const uniform = () => {
    // mulberry32
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | null = null;
  const normal = () => {
    // Box–Muller, keeping the second value
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0;
    while (u === 0) u = uniform();
    const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * uniform();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
  const gamma = (shape: number): number => {
    // Marsaglia–Tsang
    if (shape < 1) return gamma(shape + 1) * Math.pow(uniform(), 1 / shape);
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      const x = normal();
      let v = 1 + c * x;
      if (v <= 0) continue;
      v = v * v * v;
      const u = uniform();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  };
  return { uniform, normal, chi2: (nu: number) => 2 * gamma(nu / 2) };
}

// ---------- small numeric helpers ----------
function quantile(xs: ArrayLike<number>, q: number): number {
  // numpy's default (linear interpolation between order statistics)
  const a = Float64Array.from(xs).sort();
  const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}
const varOf = (r: ArrayLike<number>) => -quantile(r, 1 - CONF);
function cvarOf(r: ArrayLike<number>): number {
  const v = varOf(r);
  let s = 0, n = 0;
  for (let i = 0; i < r.length; i++) if (r[i] <= -v) { s += r[i]; n++; }
  return -s / n;
}
function ruCvar(port: ArrayLike<number>): number {
  // R-U formula evaluated at its minimiser a = VaR of losses
  const loss = Array.from(port, (x) => -x);
  const a = quantile(loss, CONF);
  let s = 0;
  for (const l of loss) s += Math.max(l - a, 0);
  return a + s / loss.length / (1 - CONF);
}
function erf(x: number): number {
  // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), enough for GELU
  const s = Math.sign(x), ax = Math.abs(x), t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return s * y;
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));
const matvec = (W: number[][], x: number[], b: number[]) => W.map((row, i) => row.reduce((s, w, j) => s + w * x[j], b[i]));

function shockFromHuman(d: EngineData, h: Shock): number[] {
  return d.factors.map((f) => {
    const v = h[HUMAN[f]] ?? 0;
    return d.log_factors.includes(f) ? Math.log1p(v / 100) : v;
  });
}
function humanFromShock(d: EngineData, s: number[]): Shock {
  const out = {} as Shock;
  d.factors.forEach((f, i) => { out[HUMAN[f]] = d.log_factors.includes(f) ? Math.expm1(s[i]) * 100 : s[i]; });
  return out;
}

// ---------- models ----------
type Mixture = { pi: number[]; mu: number[][]; sd: number[][] };

function mdnMixture(d: EngineData, shock: number[], regime: number[]): Mixture {
  const m = d.mdn;
  const x = [...shock, ...regime].map((v, i) => (v - m.x_mean[i]) / m.x_std[i]);
  const h1 = matvec(m.w1, x, m.b1).map(gelu);
  const h2 = matvec(m.w2, h1, m.b2).map(gelu);
  const out = matvec(m.w3, h2, m.b3);
  const K = m.k, D = m.assets.length;
  const logits = out.slice(0, K), mx = Math.max(...logits);
  const ex = logits.map((l) => Math.exp(l - mx)), z = ex.reduce((a, b) => a + b, 0);
  const pi = ex.map((e) => e / z); // softmax
  const mu: number[][] = [], sd: number[][] = [];
  for (let k = 0; k < K; k++) {
    mu.push([]); sd.push([]);
    for (let j = 0; j < D; j++) {
      const ls = Math.min(m.log_sig_max, Math.max(m.log_sig_min, out[K + K * D + k * D + j]));
      mu[k].push(out[K + k * D + j] * m.y_std[j] + m.y_mean[j]);
      sd[k].push(Math.exp(ls) * m.y_std[j]);
    }
  }
  return { pi, mu, sd };
}

function sampleMixture(mix: Mixture, n: number, r: ReturnType<typeof rng>): number[][] {
  const cum: number[] = [];
  mix.pi.reduce((s, p, i) => (cum[i] = s + p), 0);
  const out: number[][] = [];
  for (let s = 0; s < n; s++) {
    const u = r.uniform();
    let k = cum.findIndex((c) => u <= c);
    if (k < 0) k = cum.length - 1;
    out.push(mix.mu[k].map((m, j) => m + mix.sd[k][j] * r.normal()));
  }
  return out;
}

function baselineMean(d: EngineData, shock: number[]) {
  return d.baseline.alpha.map((a, i) => a + d.baseline.beta[i].reduce((s, b, j) => s + b * shock[j], 0));
}

function baselineSim(d: EngineData, shock: number[], n: number, r: ReturnType<typeof rng>, noMean = false): number[][] {
  // multivariate t = mean + L z / sqrt(chi2_nu / nu)
  const { chol: L, nu } = d.baseline, D = L.length;
  const mean = noMean ? new Array(D).fill(0) : baselineMean(d, shock);
  const out: number[][] = [];
  for (let s = 0; s < n; s++) {
    const z = Array.from({ length: D }, () => r.normal());
    const scale = 1 / Math.sqrt(r.chi2(nu) / nu);
    const row = new Array(D);
    for (let i = 0; i < D; i++) {
      let acc = 0;
      for (let j = 0; j <= i; j++) acc += L[i][j] * z[j];
      row[i] = mean[i] + acc * scale;
    }
    out.push(row);
  }
  return out;
}

const assetsOf = (d: EngineData) => d.mdn.assets.filter((a) => d.baseline.assets.includes(a));
const idxIn = (list: string[], assets: string[]) => assets.map((a) => list.indexOf(a));

function stressScen(d: EngineData, model: ModelKey, shock: number[], regime: number[], assets: string[], n: number, seed = SEED) {
  const r = rng(seed);
  const raw = model === "mdn" ? sampleMixture(mdnMixture(d, shock, regime), n, r) : baselineSim(d, shock, n, r);
  const idx = idxIn(model === "mdn" ? d.mdn.assets : d.baseline.assets, assets);
  return raw.map((row) => idx.map((i) => Math.expm1(row[i]))); // simple returns
}

function baseScen(d: EngineData, model: ModelKey, regime: number[], assets: string[], n: number, seed = SEED) {
  // unconditional: bootstrap past 10-day factor moves, map each through the model
  const r = rng(seed + 1);
  const shocks = Array.from({ length: n }, () => d.shock_hist[Math.floor(r.uniform() * d.shock_hist.length)]);
  let raw: number[][];
  if (model === "mdn") {
    raw = shocks.map((s) => sampleMixture(mdnMixture(d, s, regime), 1, r)[0]);
  } else {
    const resid = baselineSim(d, [], n, r, true);
    raw = shocks.map((s, k) => baselineMean(d, s).map((m, i) => m + resid[k][i]));
  }
  const idx = idxIn(model === "mdn" ? d.mdn.assets : d.baseline.assets, assets);
  return raw.map((row) => idx.map((i) => Math.expm1(row[i])));
}

function meanLog(d: EngineData, model: ModelKey, shock: number[], regime: number[], assets: string[]) {
  if (model === "mdn") {
    const mix = mdnMixture(d, shock, regime);
    const full = mix.mu[0].map((_, j) => mix.pi.reduce((s, p, k) => s + p * mix.mu[k][j], 0));
    return idxIn(d.mdn.assets, assets).map((i) => full[i]);
  }
  const m = baselineMean(d, shock);
  return idxIn(d.baseline.assets, assets).map((i) => m[i]);
}

function histogram(samples: Record<string, number[]>) {
  const pooled = Object.values(samples).flat();
  const lo = quantile(pooled, 0.002), hi = quantile(pooled, 0.998);
  const edges = Array.from({ length: N_BINS + 1 }, (_, i) => lo + ((hi - lo) * i) / N_BINS);
  const dens: Record<string, number[]> = {};
  for (const [k, v] of Object.entries(samples)) {
    const c = new Array(N_BINS).fill(0);
    for (const x of v) {
      if (x < lo || x > hi) continue; // far tails not drawn (no fake edge spikes)
      c[Math.min(N_BINS - 1, Math.floor(((x - lo) / (hi - lo)) * N_BINS))]++;
    }
    dens[k] = c.map((n) => n / v.length);
  }
  return { edges, dens };
}

function portfolio(d: EngineData, weights: Record<string, number>) {
  const all = assetsOf(d);
  const unknown = Object.keys(weights).filter((a) => !all.includes(a));
  if (unknown.length) throw new Error(`unknown tickers: ${unknown.join(", ")}`);
  const assets = all.filter((a) => (weights[a] ?? 0) > 0);
  if (!assets.length) throw new Error("portfolio has no positive weights");
  const tot = assets.reduce((s, a) => s + weights[a], 0);
  return { assets, w: assets.map((a) => weights[a] / tot) };
}

// ---------- public: same shapes as the FastAPI endpoints ----------
export async function stress(weights: Record<string, number>, human: Shock, regimeIn: Regime | null): Promise<StressResponse> {
  const d = await loadData();
  const { assets, w } = portfolio(d, weights);
  const shock = shockFromHuman(d, human);
  const regime = regimeIn ? [regimeIn.vix_level, regimeIn.avg_corr] : d.regime;
  const ports: Record<string, number[]> = {};
  const models = {} as Record<ModelKey, Omit<ModelStress, "histogram"> & { histogram?: number[] }>;
  for (const name of ["baseline", "mdn"] as ModelKey[]) {
    const scen = stressScen(d, name, shock, regime, assets, N_STRESS);
    const port = scen.map((row) => row.reduce((s, x, i) => s + x * w[i], 0));
    ports[name] = port;
    const v = varOf(port);
    const tail = scen.filter((_, s) => port[s] <= -v);
    const contrib = w.map((wi, i) => -tail.reduce((s, row) => s + row[i] * wi, 0) / tail.length);
    const meanSimple = w.map((_, i) => scen.reduce((s, row) => s + row[i], 0) / scen.length);
    // factor attribution on expected log returns (exact for baseline, one-at-a-time for MDN)
    const zero = new Array(d.factors.length).fill(0);
    const baseMean = meanLog(d, name, zero, regime, assets);
    const total = meanLog(d, name, shock, regime, assets);
    const bIdx = idxIn(d.baseline.assets, assets);
    const mat = assets.map(() => new Array(d.factors.length).fill(0));
    d.factors.forEach((_, j) => {
      let col: number[];
      if (name === "baseline") col = bIdx.map((i) => d.baseline.beta[i][j] * shock[j]);
      else {
        const one = zero.slice();
        one[j] = shock[j];
        const m1 = meanLog(d, name, one, regime, assets);
        col = m1.map((x, i) => x - baseMean[i]);
      }
      col.forEach((c, i) => (mat[i][j] = c * w[i]));
    });
    const matSum = mat.flat().reduce((a, b) => a + b, 0);
    const carry = w.reduce((s, wi, i) => s + wi * baseMean[i], 0);
    const interaction = w.reduce((s, wi, i) => s + wi * (total[i] - baseMean[i]), 0) - matSum;
    const contributions = assets.map((a, i) => ({ asset: a, weight: w[i], cvar_contrib: contrib[i], expected_contrib: w[i] * meanSimple[i] }));
    models[name] = {
      expected: port.reduce((a, b) => a + b, 0) / port.length,
      var95: v, cvar95: cvarOf(port),
      contributions,
      factor_attribution: [
        ...d.factors.map((f, j) => ({ factor: f, contribution: mat.reduce((s, row) => s + row[j], 0) })),
        { factor: "carry", contribution: carry }, { factor: "interaction", contribution: interaction },
      ],
      asset_factor: assets.map((a, i) => ({ asset: a, ...Object.fromEntries(d.factors.map((f, j) => [f, mat[i][j]])) }) as { asset: string } & Record<string, number>),
      biggest_drag: contributions.reduce((m, c) => (c.cvar_contrib > m.cvar_contrib ? c : m)).asset,
    };
  }
  const { edges, dens } = histogram(ports);
  for (const k of Object.keys(models) as ModelKey[]) models[k].histogram = dens[k];
  return { bin_edges: edges, models: models as Record<ModelKey, ModelStress>, shock: humanFromShock(d, shock), regime: { vix_level: regime[0], avg_corr: regime[1] } };
}

// ---------- CVaR LP (HiGHS / WebAssembly) ----------
type Highs = { solve: (lp: string) => { Status: string; Columns: Record<string, { Primal: number }> } };
let highsPromise: Promise<Highs> | null = null;
const loadHighs = () =>
  (highsPromise ??= import("highs").then((mod) => mod.default({ locateFile: (f: string) => `${BASE}/${f}` }) as unknown as Promise<Highs>));

const num = (x: number) => (Math.abs(x) < 1e-12 ? "0" : x.toPrecision(10));
function linear(terms: [number, string][]): string {
  // "+ 0.1 w0 - 0.2 w1 ..." with line breaks so lines stay short
  return terms
    .filter(([c]) => c !== 0)
    .map(([c, v], i) => `${c < 0 ? "-" : i ? "+" : ""} ${num(Math.abs(c))} ${v}${i % 8 === 7 ? "\n" : ""}`)
    .join(" ");
}

type LpOpts = { minRet: number | null; wMax: number; w0?: number[]; cvarCap?: number };

function buildLp(risk: number[][], mu: number[], o: LpOpts): string {
  const S = risk.length, N = mu.length, c = 1 / ((1 - CONF) * S);
  const W = Array.from({ length: N }, (_, i) => `w${i}`);
  const U = Array.from({ length: S }, (_, s) => `u${s}`);
  const cvarTerms: [number, string][] = [[1, "a"], ...U.map((u) => [c, u] as [number, string])];
  const lines: string[] = [];
  if (o.cvarCap === undefined) lines.push("Minimize", ` obj: ${linear(cvarTerms)}`);
  else lines.push("Minimize", ` obj: ${linear(W.map((_, i) => [1, `t${i}`] as [number, string]))}`);
  lines.push("Subject To");
  // u_s >= -r_s·w - a   <=>   u_s + r_s·w + a >= 0
  risk.forEach((row, s) => lines.push(` s${s}: ${linear([[1, U[s]], ...row.map((r, i) => [r, W[i]] as [number, string]), [1, "a"]])} >= 0`));
  lines.push(` budget: ${linear(W.map((w) => [1, w] as [number, string]))} = 1`);
  if (o.minRet !== null) lines.push(` ret: ${linear(mu.map((m, i) => [m, W[i]] as [number, string]))} >= ${num(o.minRet)}`);
  if (o.cvarCap !== undefined && o.w0) {
    lines.push(` cap: ${linear(cvarTerms)} <= ${num(o.cvarCap + 1e-9)}`);
    o.w0.forEach((x, i) => {
      lines.push(` p${i}: t${i} - w${i} >= ${num(-x)}`, ` q${i}: t${i} + w${i} >= ${num(x)}`);
    });
  }
  lines.push("Bounds", " -inf <= a <= inf", ...W.map((w) => ` 0 <= ${w} <= ${o.wMax}`), "End");
  return lines.join("\n");
}

async function solve(risk: number[][], mu: number[], o: LpOpts): Promise<{ w: number[] | null; status: string }> {
  const h = await loadHighs();
  const res = h.solve(buildLp(risk, mu, o));
  if (res.Status !== "Optimal") return { w: null, status: res.Status };
  let w = mu.map((_, i) => Math.max(0, res.Columns[`w${i}`]?.Primal ?? 0));
  const s = w.reduce((a, b) => a + b, 0);
  w = w.map((x) => x / s);
  return { w, status: "optimal" };
}

export async function optimize(body: { weights: Record<string, number>; shock: Shock; model: ModelKey; mode: string }): Promise<OptimizeResponse> {
  const d = await loadData();
  const held = portfolio(d, body.weights);
  const assets = assetsOf(d); // the optimiser may buy anything in the universe
  const w0 = assets.map((a) => { const k = held.assets.indexOf(a); return k < 0 ? 0 : held.w[k]; });
  const shock = shockFromHuman(d, body.shock);
  const risk = stressScen(d, body.model, shock, d.regime, assets, N_OPT);
  const ret = baseScen(d, body.model, d.regime, assets, N_OPT);
  const mu = assets.map((_, i) => ret.reduce((s, row) => s + row[i], 0) / ret.length);
  const wMax = 0.15, minRet = 0;
  const pr = (w: number[]) => risk.map((row) => row.reduce((s, x, i) => s + x * w[i], 0));
  const er = (w: number[]) => mu.reduce((s, m, i) => s + m * w[i], 0);

  let full = await solve(risk, mu, { minRet, wMax });
  let status = full.status;
  if (!full.w) { full = await solve(risk, mu, { minRet: null, wMax }); status = "return floor infeasible; dropped"; }
  const fullW = full.w!;
  let w = fullW;
  if (body.mode === "minimal_change") {
    const c0 = ruCvar(pr(w0)), cStar = ruCvar(pr(fullW));
    const small = await solve(risk, mu, { minRet: status === "optimal" ? minRet : null, wMax, w0, cvarCap: c0 - 0.8 * (c0 - cStar) });
    if (small.w) w = small.w;
  }
  const before = pr(w0), after = pr(w);
  const { edges, dens } = histogram({ before, after });
  // frontier: sweep the return floor from the min-CVaR portfolio's return to the max under the caps
  const lo = er(fullW);
  const order = mu.map((m, i) => [m, i]).sort((a, b) => b[0] - a[0]);
  let left = 1, hi = 0;
  for (const [m] of order) { const x = Math.min(wMax, left); hi += m * x; left -= x; }
  const frontier: { exp_ret: number; cvar: number }[] = [];
  for (let k = 0; k < 10; k++) {
    const target = lo + ((hi - lo) * k) / 9;
    const r = await solve(risk, mu, { minRet: target - 1e-9, wMax });
    if (r.w) frontier.push({ exp_ret: er(r.w), cvar: ruCvar(pr(r.w)) });
  }
  const turnover = w.reduce((s, x, i) => s + Math.abs(x - w0[i]), 0);
  return {
    status, mode: body.mode, model: body.model, turnover,
    weights: Object.fromEntries(assets.map((a, i) => [a, w[i]]).filter(([, x]) => (x as number) > 5e-4)),
    trades: assets.map((a, i) => ({ asset: a, from: w0[i], to: w[i], delta: w[i] - w0[i] }))
      .filter((t) => Math.abs(t.delta) > 5e-4).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)),
    bin_edges: edges,
    before: { histogram: dens.before, var95: varOf(before), cvar95: ruCvar(before), expected_stress: before.reduce((a, b) => a + b, 0) / before.length, expected_base: er(w0) },
    after: { histogram: dens.after, var95: varOf(after), cvar95: ruCvar(after), expected_stress: after.reduce((a, b) => a + b, 0) / after.length, expected_base: er(w) },
    full_optimum_cvar95: ruCvar(pr(fullW)),
    frontier,
  };
}
