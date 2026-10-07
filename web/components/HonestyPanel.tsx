"use client";
import type { ComparisonRow } from "@/lib/api";
import { pct } from "@/lib/format";
import { Panel } from "./ui";

type Col = { key: string; label: string; fmt: (v: number) => string; score: (v: number) => number; hint: string };

// score: lower is better (distance from the ideal)
const COLS: Col[] = [
  { key: "mean_nll", label: "Mean NLL", fmt: (v) => v.toFixed(1), score: (v) => v, hint: "joint −log density of the 24 realized returns; lower = better" },
  { key: "asset_var_breach_rate", label: "Asset VaR breach", fmt: (v) => pct(v, 1), score: (v) => Math.abs(v - 0.05), hint: "target 5%" },
  { key: "port_var_breach_rate", label: "Portfolio VaR breach", fmt: (v) => pct(v, 1), score: (v) => Math.abs(v - 0.05), hint: "target 5%" },
  { key: "kupiec_p", label: "Kupiec p", fmt: (v) => v.toFixed(2), score: (v) => -v, hint: "p < 0.05 rejects a correct 5% breach rate" },
  { key: "port_cvar_error", label: "CVaR error", fmt: (v) => `${v >= 0 ? "+" : "−"}${pct(Math.abs(v), 2)}`, score: (v) => Math.abs(v), hint: "realized loss − predicted CVaR on breach windows; 0 is ideal" },
  { key: "port_mean_abs_error", label: "Portfolio MAE", fmt: (v) => pct(v, 2), score: (v) => v, hint: "|realized − predicted mean|" },
];

const MODEL_COLOR: Record<string, string> = {
  "Baseline (OLS + t)": "var(--base)", "MDN static (train<=2017)": "var(--mdn)", "MDN walk-forward": "var(--mdn)",
};

export default function HonestyPanel({ rows }: { rows: ComparisonRow[] }) {
  const test = rows.filter((r) => r.scope === "test_2020+");
  const events = rows.filter((r) => r.scope === "event");
  const best = Object.fromEntries(COLS.map((c) => {
    const scores = test.map((r) => c.score(Number(r[c.key])));
    return [c.key, Math.min(...scores)];
  }));
  const base = test.find((r) => String(r.model).startsWith("Baseline"));
  const wf = test.find((r) => r.model === "MDN walk-forward");
  const nllGap = base && wf ? Number(wf.mean_nll) - Number(base.mean_nll) : 0;
  const evWins = new Set(events.map((e) => e.event)).size;
  const nllEventWinsBase = [...new Set(events.map((e) => e.event))].filter((name) => {
    const evr = events.filter((e) => e.event === name);
    const b = evr.find((e) => String(e.model).startsWith("Baseline"));
    return b && evr.every((e) => Number(e.nll) >= Number(b.nll));
  }).length;

  return (
    <Panel kicker="08" title="Model honesty" right={<span className="text-[11px] text-muted">out-of-sample, 2020 → today · 169 non-overlapping 10-day windows</span>}>
      <p className="mb-4 max-w-3xl text-sm leading-relaxed text-ink-2">
        <span className="text-ink">{nllGap > 0 ? "The MDN does not beat the linear baseline." : "The MDN beats the baseline on likelihood."}</span>{" "}
        Baseline NLL is {nllGap > 0 ? "better" : "worse"} by <span className="num text-ink">{Math.abs(nllGap).toFixed(1)}</span> nats per window
        vs. the walk-forward MDN, and the baseline has the lowest NLL in{" "}
        <span className="num text-ink">{nllEventWinsBase}/{evWins}</span> stress events.
        {nllGap > 0 && " A full residual covariance captures cross-asset co-movement better than five diagonal mixture components."}
        {" "}Columns marked BEST show where each model actually wins.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-xs">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              <th className="py-2 pr-3 font-normal">Model</th>
              {COLS.map((c) => <th key={c.key} className="py-2 pr-3 text-right font-normal" title={c.hint}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {test.map((r) => (
              <tr key={String(r.model)} className="border-b border-line/60">
                <td className="py-2 pr-3 text-ink">
                  <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: MODEL_COLOR[String(r.model)] }} />{String(r.model)}
                </td>
                {COLS.map((c) => {
                  const v = Number(r[c.key]);
                  const win = Math.abs(c.score(v) - best[c.key]) < 1e-12;
                  return (
                    <td key={c.key} className={`num py-2 pr-3 text-right ${win ? "text-ink" : "text-muted"}`}>
                      {c.fmt(v)}{win && <span className="ml-1 text-[9px] text-amber">BEST</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-5 text-[11px] uppercase tracking-widest text-muted">Stress events (equal-weight portfolio)</div>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[640px] text-xs">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              {["Event", "Model", "NLL", "Asset misses", "Pred. mean", "VaR 95", "CVaR 95", "Realized", "CVaR error"].map((h, i) => (
                <th key={h} className={`py-2 pr-3 font-normal ${i > 1 ? "text-right" : ""}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {events.map((r, i) => {
              const breach = r.port_breach === true || r.port_breach === "True";
              return (
                <tr key={i} className={`border-b border-line/40 ${i % 3 === 0 ? "border-t border-t-line" : ""}`}>
                  <td className="py-1.5 pr-3 text-ink-2">{i % 3 === 0 ? String(r.event) : ""}</td>
                  <td className="py-1.5 pr-3 text-ink-2">{String(r.model).replace(" (OLS + t)", "").replace(" (train<=2017)", "")}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink">{Number(r.nll).toFixed(1)}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink-2">{String(r.asset_breaches)}/{String(r.n_assets)}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink-2">{pct(Number(r.pred_port_mean), 2)}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink-2">{pct(Number(r.pred_var95), 2)}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink-2">{pct(Number(r.pred_cvar95), 2)}</td>
                  <td className={`num py-1.5 pr-3 text-right ${breach ? "text-loss" : "text-ink-2"}`}>{pct(Number(r.realized_port), 2)}{breach ? " ✕" : ""}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink-2">{pct(Number(r.cvar_error), 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-muted">✕ = realized loss exceeded the model&apos;s VaR 95. CVaR error = realized loss − predicted CVaR.</p>
    </Panel>
  );
}
