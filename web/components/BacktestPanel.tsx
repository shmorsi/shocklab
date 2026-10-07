"use client";
import * as d3 from "d3";
import { useState } from "react";

import type { Backtest } from "@/lib/api";
import { pct } from "@/lib/format";
import { useWidth } from "@/lib/hooks";
import { Panel } from "./ui";

// The two optimised strategies carry colour; benchmarks are recessive grey context lines.
const STYLE: Record<string, { color: string; dash?: string; width: number }> = {
  "CVaR (MDN)": { color: "var(--mdn)", width: 2 },
  "CVaR (baseline)": { color: "var(--base)", width: 2 },
  "Min variance": { color: "#9aa1ab", width: 1.5 },
  "60/40 SPY/IEF": { color: "#7b828c", dash: "5 3", width: 1.5 },
  "Equal weight": { color: "#7b828c", dash: "1.5 3", width: 1.5 },
};

export default function BacktestPanel({ bt }: { bt: Backtest }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hi, setHi] = useState<number | null>(null);
  const names = Object.keys(bt.equity);
  const dates = bt.dates.map((d) => new Date(d));
  const h = 260, m = { t: 12, r: 96, b: 26, l: 40 };
  const w = Math.max(width, 280);
  const x = d3.scaleTime().domain(d3.extent(dates) as [Date, Date]).range([m.l, w - m.r]);
  const all = names.flatMap((n) => bt.equity[n]);
  const [lo, hiV] = d3.extent(all) as [number, number];
  const y = d3.scaleLog().domain([lo * 0.97, hiV * 1.03]).range([h - m.b, m.t]);
  const yTicks = [0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4].filter((t) => t >= lo * 0.97 && t <= hiV * 1.03);
  const line = (vals: number[]) => d3.line<number>().x((_, i) => x(dates[i])).y((v) => y(v))(vals) ?? "";
  // spread end labels so they don't overlap
  const ends = names.map((n) => ({ n, y: y(bt.equity[n][bt.equity[n].length - 1]) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 12);

  return (
    <Panel kicker="09" title="Walk-forward backtest" right={<span className="text-[11px] text-muted">monthly rebalance from 2018 · 10 bp costs · growth of $1 (log scale)</span>}>
      <div ref={ref} className="relative w-full" style={{ height: h }}
        onMouseMove={(e) => {
          const r = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
          const d = x.invert(e.clientX - r.left);
          setHi(Math.max(0, Math.min(dates.length - 1, d3.bisectLeft(dates, d))));
        }}
        onMouseLeave={() => setHi(null)}>
        {width > 0 && (
          <svg width={w} height={h} role="img" aria-label="Equity curves of backtested strategies">
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                <text x={m.l - 6} y={y(t) + 3} textAnchor="end" fontSize={10} className="num" fill="var(--muted)">{t.toFixed(2).replace(/0$/, "")}</text>
              </g>
            ))}
            {x.ticks(w < 500 ? 4 : 8).map((t) => (
              <text key={+t} x={x(t)} y={h - 8} textAnchor="middle" fontSize={10} className="num" fill="var(--muted)">{t.getFullYear()}</text>
            ))}
            {[...names].reverse().map((n) => (
              <path key={n} d={line(bt.equity[n])} fill="none" stroke={STYLE[n]?.color} strokeWidth={STYLE[n]?.width} strokeDasharray={STYLE[n]?.dash} />
            ))}
            {ends.map((e) => (
              <text key={e.n} x={w - m.r + 6} y={e.y + 3} fontSize={10} fill="var(--ink-2)">{e.n}</text>
            ))}
            {hi !== null && <line x1={x(dates[hi])} x2={x(dates[hi])} y1={m.t} y2={h - m.b} stroke="var(--muted)" />}
          </svg>
        )}
        {hi !== null && (
          <div className="pointer-events-none absolute top-2 z-10 rounded border border-line bg-bg/95 px-2 py-1.5 text-[11px]"
            style={{ left: Math.min(x(dates[hi]) + 10, w - 190) }}>
            <div className="num mb-1 text-muted">{bt.dates[hi]}</div>
            {names.map((n) => (
              <div key={n} className="flex justify-between gap-4">
                <span className="flex items-center gap-1.5 text-ink-2"><span className="h-0.5 w-3" style={{ background: STYLE[n]?.color }} />{n}</span>
                <span className="num text-ink">{bt.equity[n][hi].toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[560px] text-xs">
          <thead>
            <tr className="border-b border-line text-left text-muted">
              {["Strategy", "CAGR", "Vol", "Max DD", "Worst month", "Sharpe", "Turnover/mo"].map((c, i) => (
                <th key={c} className={`py-2 pr-3 font-normal ${i ? "text-right" : ""}`}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {bt.metrics.map((r) => (
              <tr key={String(r.strategy)} className="border-b border-line/50">
                <td className="py-1.5 pr-3 text-ink">
                  <span className="mr-2 inline-block h-0.5 w-3 align-middle" style={{ background: STYLE[String(r.strategy)]?.color }} />{String(r.strategy)}
                </td>
                {["CAGR", "Volatility", "Max drawdown", "Worst month"].map((k) => (
                  <td key={k} className="num py-1.5 pr-3 text-right text-ink-2">{pct(Number(r[k]), 1)}</td>
                ))}
                <td className="num py-1.5 pr-3 text-right text-ink">{Number(r.Sharpe).toFixed(2)}</td>
                <td className="num py-1.5 pr-3 text-right text-ink-2">{pct(Number(r["Avg monthly turnover"]), 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-muted">{verdict(bt)} TSX names held in CAD.</p>
    </Panel>
  );
}

/** One-line summary computed from the metrics, so the copy can never drift from the data. */
function verdict(bt: Backtest): string {
  const get = (n: string) => bt.metrics.find((r) => r.strategy === n);
  const cv = get("CVaR (MDN)"), mv = get("Min variance"), ew = get("Equal weight");
  if (!cv || !mv || !ew) return "";
  const dd = Number(cv["Max drawdown"]) / Number(ew["Max drawdown"]);
  const beat = Number(cv.Sharpe) > Number(mv.Sharpe);
  return `CVaR (MDN) had ${pct(dd, 0)} of equal weight's max drawdown and ${beat ? "beat" : "did not beat"} minimum variance on Sharpe (${Number(cv.Sharpe).toFixed(2)} vs ${Number(mv.Sharpe).toFixed(2)}).`;
}
