"use client";
import { useEffect, useRef, useState } from "react";

import { api, type ModelKey, type OptimizeResponse, type Shock } from "@/lib/api";
import { pct, spct } from "@/lib/format";
import DivergingBars from "./DivergingBars";
import Frontier from "./Frontier";
import Histogram from "./Histogram";
import { Panel, Segmented } from "./ui";

export default function OptimizerPanel({ weights, shock, model }: { weights: Record<string, number>; shock: Shock; model: ModelKey }) {
  const [mode, setMode] = useState<"min_cvar" | "minimal_change">("min_cvar");
  const [res, setRes] = useState<OptimizeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // run once shortly after load so the panel isn't empty; afterwards only on demand (the LP takes ~1-2 s)
  const ran = useRef(false);
  useEffect(() => {
    if (ran.current || !Object.keys(weights).length) return;
    // mark as run inside the timer: StrictMode's mount/unmount/mount would otherwise cancel it
    const id = setTimeout(() => { ran.current = true; run(); }, 600);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weights]);

  async function run() {
    setBusy(true);
    setErr(null);
    try {
      setRes(await api.optimize({ weights, shock, model, mode }));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel kicker="06" title="CVaR optimiser"
      right={
        <div className="flex flex-wrap items-center gap-2">
          <Segmented label="Optimiser mode" value={mode} onChange={setMode}
            options={[{ value: "min_cvar", label: "Min CVaR" }, { value: "minimal_change", label: "Minimal change" }]} />
          <button onClick={run} disabled={busy}
            className="rounded border border-amber bg-amber/10 px-3 py-1 text-xs text-amber transition hover:bg-amber/20 disabled:opacity-50">
            {busy ? "solving LP…" : "Optimise for this shock"}
          </button>
        </div>
      }>
      <p className="mb-3 text-[11px] leading-relaxed text-muted">
        Minimises CVaR 95 over {model === "mdn" ? "MDN" : "baseline"} scenarios under the current shock (Rockafellar–Uryasev LP),
        long-only, max 15% per asset, expected return ≥ 0 under normal (bootstrapped) macro conditions.
        {mode === "minimal_change" && " Minimal change: smallest turnover that captures 80% of the possible CVaR reduction."}
      </p>
      {err && <p className="num text-xs text-loss">{err}</p>}
      {!res && !err && <div className="grid h-40 place-items-center rounded border border-dashed border-line text-xs text-muted">Set a shock, then optimise.</div>}
      {res && (
        <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr]">
          <div>
            <div className="mb-1 flex flex-wrap gap-x-5 gap-y-1 text-xs">
              <span className="text-muted">CVaR <span className="num text-ink">{pct(res.before.cvar95, 2)} → {pct(res.after.cvar95, 2)}</span></span>
              <span className="text-muted">E[r] stressed <span className="num text-ink">{spct(res.before.expected_stress, 2)} → {spct(res.after.expected_stress, 2)}</span></span>
              <span className="text-muted">E[r] normal <span className="num text-ink">{spct(res.before.expected_base, 2)} → {spct(res.after.expected_base, 2)}</span></span>
              <span className="text-muted">turnover <span className="num text-ink">{pct(res.turnover, 0)}</span></span>
            </div>
            <Histogram edges={res.bin_edges} density={res.after.histogram} ghost={res.before.histogram} height={200}
              lines={[{ x: -res.after.var95, label: "VaR", dash: true }, { x: -res.after.cvar95, label: "CVaR" }]} />
            <div className="mt-1 text-[11px] text-muted">bars: suggested portfolio · dashed outline: current portfolio</div>
            <div className="mt-4 text-[11px] uppercase tracking-widest text-muted">Frontier</div>
            <Frontier points={res.frontier}
              before={{ cvar: res.before.cvar95, exp_ret: res.before.expected_base }}
              after={{ cvar: res.after.cvar95, exp_ret: res.after.expected_base }} />
          </div>
          <div>
            <div className="mb-2 text-[11px] uppercase tracking-widest text-muted">Suggested trades</div>
            <DivergingBars rows={res.trades.map((t) => ({ label: t.asset, value: t.delta, note: `${pct(t.from, 1)} → ${pct(t.to, 1)}` }))}
              fmt={(x) => spct(x, 1)} maxRows={14} />
            {res.status !== "optimal" && <p className="mt-2 text-[11px] text-amber">solver: {res.status}</p>}
          </div>
        </div>
      )}
    </Panel>
  );
}
