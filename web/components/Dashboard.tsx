"use client";
import { useEffect, useMemo, useRef, useState } from "react";

import { api, STATIC, type Backtest, type ComparisonRow, type EventReplay as Ev, type Meta, type ModelKey, type Regime, type Shock, type StressResponse } from "@/lib/api";
import { pct } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import BacktestPanel from "./BacktestPanel";
import EventReplay from "./EventReplay";
import Histogram from "./Histogram";
import HonestyPanel from "./HonestyPanel";
import KpiTiles from "./KpiTiles";
import OptimizerPanel from "./OptimizerPanel";
import PortfolioEditor, { presetWeights } from "./PortfolioEditor";
import Propagation from "./Propagation";
import RiskAttribution from "./RiskAttribution";
import ShockPanel, { clampShock, ZERO_SHOCK } from "./ShockPanel";
import { MODEL_OPTS, Panel, Segmented } from "./ui";

export default function Dashboard() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [events, setEvents] = useState<Ev[]>([]);
  const [comparison, setComparison] = useState<ComparisonRow[]>([]);
  const [bt, setBt] = useState<Backtest | null>(null);
  const [bootErr, setBootErr] = useState<string | null>(null);

  const [weights, setWeights] = useState<Record<string, number>>({});
  const [shock, setShock] = useState<Shock>({ ...ZERO_SHOCK, oil_pct: -25, vix_pct: 80, credit_bp: 60 });
  const [preset, setPreset] = useState<string | null>(null);
  const [regime, setRegime] = useState<{ value: Regime; label: string } | null>(null);
  const [model, setModel] = useState<ModelKey>("mdn");
  const [stress, setStress] = useState<StressResponse | null>(null);
  const [doneQuery, setDoneQuery] = useState<object | null>(null);

  useEffect(() => {
    Promise.all([api.meta(), api.events(), api.comparison(), api.backtest()])
      .then(([m, e, c, b]) => {
        setMeta(m); setEvents(e); setComparison(c); setBt(b);
        setWeights(presetWeights("Equal weight", m.assets));
      })
      .catch((e) => setBootErr(String(e)));
  }, []);

  // debounce slider + portfolio changes; abort stale requests
  const query = useDebounced(useMemo(() => ({ weights, shock, regime: regime?.value ?? null }), [weights, shock, regime]), 200);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!Object.keys(query.weights).length) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    api.stress(query.weights, query.shock, query.regime, ac.signal)
      .then((r) => { setStress(r); setDoneQuery(query); })
      .catch((e) => { if (e.name !== "AbortError") { setBootErr(String(e)); setDoneQuery(query); } });
  }, [query]);
  const loading = doneQuery !== query;

  const applyPreset = (e: Ev | null) => {
    if (!e) { setPreset(null); setShock(ZERO_SHOCK); setRegime(null); return; }
    setPreset(e.name);
    setShock(clampShock(e.shock));
    setRegime({ value: e.regime, label: `${e.name} (${e.start})` }); // replay with that day's regime
  };
  const onShock = (s: Shock) => { setShock(s); setPreset(null); };

  if (bootErr && !meta) {
    return (
      <main className="relative z-10 mx-auto grid min-h-screen max-w-xl place-items-center px-4">
        <div className="rounded-md border border-line bg-panel p-6 text-sm">
          <div className="num mb-2 text-amber">{STATIC ? "Could not load model data" : "API unreachable"}</div>
          <p className="text-ink-2">{STATIC ? "Reload the page; if it persists, the data files failed to download." : <>Start the API with <code className="num text-ink">make api</code> (port 8000), then reload.</>}</p>
          <p className="num mt-3 break-all text-[11px] text-muted">{bootErr}</p>
        </div>
      </main>
    );
  }

  const m = stress?.models[model];
  const other = stress?.models[model === "mdn" ? "baseline" : "mdn"];
  return (
    <main className="relative z-10 mx-auto max-w-[1400px] px-4 pb-16 pt-6 sm:px-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="num text-[11px] tracking-[0.3em] text-amber">SHOCK/LAB</div>
          <h1 className="mt-1 text-2xl font-medium tracking-tight sm:text-3xl">What does a macro shock do to this portfolio?</h1>
          <p className="mt-1 max-w-2xl text-sm text-ink-2">
            10-day return distribution of a 24-asset portfolio under oil, rates, USD, VIX and credit shocks. Mixture density network vs.
            a linear factor + Student-t baseline, scored honestly out of sample.
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <Segmented label="Model" value={model} onChange={setModel} options={MODEL_OPTS} />
          {meta && <span className="num text-[11px] text-muted">data to {meta.as_of} · MDN trained through {meta.mdn_trained_through}{STATIC ? " · models run in your browser" : ""}</span>}
        </div>
      </header>

      <div className="mb-4">
        {meta && <PortfolioEditor assets={meta.assets} weights={weights} setWeights={setWeights} />}
      </div>

      <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
        <div className="space-y-4 lg:sticky lg:top-4 lg:self-start">
          <ShockPanel shock={shock} setShock={onShock} events={events} activePreset={preset} onPreset={applyPreset}
            regime={regime?.value ?? meta?.regime ?? null} regimeLabel={regime?.label ?? "today"} onResetRegime={() => setRegime(null)} range={meta?.shock_range} />
          <p className="px-1 text-[11px] leading-relaxed text-muted">
            Credit uses Moody&apos;s Baa−10Y spread: FRED now serves only 3 years of the ICE HY OAS. Shocks are 10-trading-day moves;
            the models were trained on realized moves of this size.
          </p>
        </div>

        <div className="min-w-0 space-y-4">
          {m && other ? (
            <>
              <KpiTiles m={m} other={other} otherLabel={model === "mdn" ? "baseline" : "MDN"} />
              <Panel kicker="03" title="10-day P&L distribution"
                right={<span className={`num text-[11px] ${loading ? "text-amber" : "text-muted"}`}>{loading ? "simulating…" : "10,000 scenarios"}</span>}>
                <Histogram edges={stress!.bin_edges} density={m.histogram}
                  lines={[{ x: -m.var95, label: "VaR", dash: true }, { x: -m.cvar95, label: "CVaR" }]} />
                <div className="mt-1 flex flex-wrap justify-between gap-2 text-[11px] text-muted">
                  <span><span className="text-loss">■</span> loss · <span className="text-gain">■</span> gain · dashed = VaR 95, solid = CVaR 95</span>
                  <span className="num">
                    {model === "mdn" ? "baseline" : "MDN"}: VaR {pct(other.var95, 2)} · CVaR {pct(other.cvar95, 2)}
                  </span>
                </div>
              </Panel>
              <Propagation m={m} model={model} />
              <RiskAttribution m={m} />
            </>
          ) : (
            <div className="grid h-96 place-items-center rounded-md border border-line bg-panel text-sm text-muted">loading models…</div>
          )}
        </div>
      </div>

      <div className="mt-4 space-y-4">
        {meta && <OptimizerPanel weights={weights} shock={shock} model={model} />}
        {events.length > 0 && <EventReplay events={events} model={model} />}
        {comparison.length > 0 && <HonestyPanel rows={comparison} />}
        {bt && <BacktestPanel bt={bt} />}
      </div>
      <footer className="mt-10 text-center text-[11px] text-muted">
        Shock Lab · research tool, not investment advice · no lookahead: every number is from models fit only on prior data
      </footer>
    </main>
  );
}
