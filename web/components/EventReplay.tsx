"use client";
import * as d3 from "d3";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

import type { EventReplay as Ev, ModelKey } from "@/lib/api";
import { bp, pct, spct } from "@/lib/format";
import { useWidth } from "@/lib/hooks";
import { Panel } from "./ui";

const MODEL_COLOR = { mdn: "var(--mdn)", baseline: "var(--base)" };

export default function EventReplay({ events, model }: { events: Ev[]; model: ModelKey }) {
  const [sel, setSel] = useState(0);
  const [day, setDay] = useState(0);
  const [playing, setPlaying] = useState(true);
  const ev = events[sel];
  const em = ev.models[model];

  // advance one trading day every 500ms while playing
  useEffect(() => {
    if (!playing || day >= ev.path.length - 1) return;
    const id = setTimeout(() => setDay((d) => d + 1), 500);
    return () => clearTimeout(id);
  }, [playing, day, ev.path.length]);

  const pick = (i: number) => { setSel(i); setDay(0); setPlaying(true); };
  const [ref, width] = useWidth<HTMLDivElement>();
  const h = 260, m = { t: 14, r: 64, b: 28, l: 48 };
  const w = Math.max(width, 260);
  const vals = [...em.band.flatMap((b) => [b.q05, b.q95]), ...ev.path.map((p) => p.value)];
  const x = d3.scaleLinear().domain([0, 10]).range([m.l, w - m.r]);
  const y = d3.scaleLinear().domain(d3.extent(vals) as [number, number]).nice().range([h - m.b, m.t]);
  const area = (lo: "q05" | "q25", hi: "q95" | "q75") =>
    d3.area<(typeof em.band)[0]>().x((b) => x(b.day)).y0((b) => y(b[lo])).y1((b) => y(b[hi]))(em.band) ?? "";
  const shown = ev.path.slice(0, day + 1);
  const pathD = d3.line<(typeof ev.path)[0]>().x((p) => x(p.day)).y((p) => y(p.value))(shown) ?? "";
  const last = shown[shown.length - 1];
  const band = em.band[last.day];
  const outside = last.value < band.q05 || last.value > band.q95;
  const s = ev.shock;

  return (
    <Panel kicker="07" title="Backtest replay"
      right={<span className="text-[11px] text-muted">fit before the event · realized factor moves fed in</span>}>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {events.map((e, i) => (
          <button key={e.name} onClick={() => pick(i)}
            className={`rounded border px-2 py-1 text-xs ${i === sel ? "border-amber text-amber" : "border-line text-ink-2 hover:text-ink"}`}>
            {e.name}
          </button>
        ))}
      </div>
      <div className="num mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-2">
        <span className="text-muted">{ev.start} → {ev.end}</span>
        <span>oil <span className="text-amber">{spct(s.oil_pct / 100)}</span></span>
        <span>10Y <span className="text-amber">{bp(s.rates_bp)}</span></span>
        <span>USD <span className="text-amber">{spct(s.usd_pct / 100)}</span></span>
        <span>VIX <span className="text-amber">{spct(s.vix_pct / 100, 0)}</span></span>
        <span>credit <span className="text-amber">{bp(s.credit_bp)}</span></span>
      </div>
      <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <div ref={ref} className="relative w-full" style={{ height: h }}>
            {width > 0 && (
              <svg width={w} height={h} role="img" aria-label={`Predicted band vs realized path, ${ev.name}`}>
                {y.ticks(5).map((t) => (
                  <g key={t}>
                    <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                    <text x={m.l - 6} y={y(t) + 3} textAnchor="end" fontSize={10} className="num" fill="var(--muted)">{pct(t, 0)}</text>
                  </g>
                ))}
                {x.ticks(10).map((t) => (
                  <text key={t} x={x(t)} y={h - 10} textAnchor="middle" fontSize={10} className="num" fill="var(--muted)">{t}</text>
                ))}
                <line x1={m.l} x2={w - m.r} y1={y(0)} y2={y(0)} stroke="var(--line)" />
                <motion.g key={`${sel}-${model}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <path d={area("q05", "q95")} fill={MODEL_COLOR[model]} fillOpacity={0.16} />
                  <path d={area("q25", "q75")} fill={MODEL_COLOR[model]} fillOpacity={0.28} />
                </motion.g>
                <path d={d3.line<(typeof em.band)[0]>().x((b) => x(b.day)).y((b) => y(b.mean))(em.band) ?? ""} stroke={MODEL_COLOR[model]} strokeDasharray="4 3" strokeWidth={1.5} fill="none" />
                <path d={pathD} fill="none" stroke="var(--ink)" strokeWidth={2} />
                {shown.map((p) => {
                  const b = em.band[p.day];
                  const out = p.value < b.q05 || p.value > b.q95;
                  return <circle key={p.day} cx={x(p.day)} cy={y(p.value)} r={p.day === last.day ? 5 : 3}
                    fill={out ? "var(--loss)" : "var(--ink)"} stroke="var(--bg)" strokeWidth={2} />;
                })}
                <text x={w - m.r + 6} y={y(em.band[10].q95) + 3} fontSize={10} fill="var(--ink-2)">95%</text>
                <text x={w - m.r + 6} y={y(em.band[10].q05) + 3} fontSize={10} fill="var(--ink-2)">5%</text>
                <text x={w - m.r + 6} y={y(em.band[10].mean) + 3} fontSize={10} fill="var(--ink-2)">pred.</text>
              </svg>
            )}
          </div>
          <div className="mt-1 flex items-center gap-3 text-xs">
            <button onClick={() => { if (day >= 10) { setDay(0); setPlaying(true); } else setPlaying(!playing); }} className="rounded border border-line px-2 py-0.5 text-ink-2 hover:text-ink">
              {playing && day < 10 ? "pause" : day >= 10 ? "replay" : "play"}
            </button>
            <input type="range" min={0} max={10} value={day} aria-label="Trading day"
              onChange={(e) => { setPlaying(false); setDay(Number(e.target.value)); }} className="max-w-48" />
            <span className="num text-muted">day {last.day} · {last.date}</span>
            <span className={`num ml-auto ${outside ? "text-loss" : "text-ink-2"}`}>
              {spct(last.value, 2)} {outside ? "outside 90% band" : "inside band"}
            </span>
          </div>
          <p className="mt-2 text-[11px] text-muted">
            Equal-weight portfolio. Models predict the 10-day horizon; intermediate days scale the centre by d/10 and the spread by √(d/10).
          </p>
        </div>
        <AssetStrip em={em} color={MODEL_COLOR[model]} summary={ev} model={model} />
      </div>
    </Panel>
  );
}

function AssetStrip({ em, color, summary, model }: { em: Ev["models"][ModelKey]; color: string; summary: Ev; model: ModelKey }) {
  const lo = Math.min(...em.assets.map((a) => Math.min(a.lo, a.realized)));
  const hi = Math.max(...em.assets.map((a) => Math.max(a.hi, a.realized)));
  const sx = (v: number) => ((v - lo) / (hi - lo)) * 100;
  const misses = em.assets.filter((a) => a.realized < a.lo || a.realized > a.hi).length;
  const breach = summary.realized_port < -em.var95;
  return (
    <div>
      <div className="mb-2 grid grid-cols-3 gap-2 text-xs">
        <div><div className="text-muted">VaR 95</div><div className="num text-ink">{pct(em.var95, 2)}</div></div>
        <div><div className="text-muted">Realized</div><div className={`num ${summary.realized_port < 0 ? "text-loss" : "text-gain"}`}>{spct(summary.realized_port, 2)}</div></div>
        <div><div className="text-muted">Breach</div><div className={`num ${breach ? "text-loss" : "text-ink"}`}>{breach ? "yes" : "no"}</div></div>
      </div>
      <div className="mb-1 text-[11px] text-muted">
        Per asset ({model === "mdn" ? "MDN" : "baseline"} 5–95% band, ● realized) · <span className="num text-ink-2">{misses}/{em.assets.length}</span> outside
      </div>
      <ul className="max-h-[300px] space-y-[3px] overflow-y-auto pr-1">
        {em.assets.map((a) => {
          const out = a.realized < a.lo || a.realized > a.hi;
          return (
            <li key={a.asset} className="grid grid-cols-[56px_1fr_52px] items-center gap-2 text-[11px]">
              <span className="num text-ink-2">{a.asset}</span>
              <span className="relative h-3">
                <span className="absolute top-[5px] h-px w-full bg-line" />
                <span className="absolute top-[3px] h-[6px] rounded-sm" style={{ left: `${sx(a.lo)}%`, width: `${sx(a.hi) - sx(a.lo)}%`, background: color, opacity: 0.55 }} />
                <span className="absolute top-[1px] h-[10px] w-px bg-ink-2" style={{ left: `${sx(a.pred)}%` }} />
                <span className="absolute top-[2px] h-2 w-2 -translate-x-1/2 rounded-full ring-2 ring-panel" style={{ left: `${sx(a.realized)}%`, background: out ? "var(--loss)" : "var(--ink)" }} />
              </span>
              <span className={`num text-right ${out ? "text-loss" : "text-ink-2"}`}>{spct(a.realized, 1)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
