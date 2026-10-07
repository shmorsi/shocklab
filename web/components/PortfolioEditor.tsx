"use client";
import { useState } from "react";

import { pct } from "@/lib/format";

export const PRESETS: Record<string, Record<string, number>> = {
  "Equal weight": {},
  "60/40": { SPY: 60, IEF: 40 },
  "Canada tilt": { "XIU.TO": 30, "RY.TO": 15, "CNQ.TO": 10, "SU.TO": 10, "CP.TO": 10, IEF: 15, GLD: 10 },
  "Tech heavy": { QQQ: 40, XLK: 25, SPY: 15, TLT: 10, GLD: 10 },
  "Energy & travel": { XLE: 25, USO: 15, "CNQ.TO": 15, JETS: 15, SPY: 30 },
};

export function presetWeights(name: string, assets: string[]): Record<string, number> {
  const p = PRESETS[name];
  if (!p || Object.keys(p).length === 0) return Object.fromEntries(assets.map((a) => [a, 1]));
  return { ...p };
}

export default function PortfolioEditor({ assets, weights, setWeights }: {
  assets: string[]; weights: Record<string, number>; setWeights: (w: Record<string, number>) => void;
}) {
  const [open, setOpen] = useState(false);
  const [preset, setPreset] = useState("Equal weight");
  const total = Object.values(weights).reduce((a, b) => a + b, 0) || 1;
  const held = assets.filter((a) => (weights[a] ?? 0) > 0);
  return (
    <div className="rounded-md border border-line bg-panel/90">
      <div className="flex flex-wrap items-center gap-2 px-4 py-2.5">
        <span className="text-[11px] uppercase tracking-widest text-muted">Portfolio</span>
        {Object.keys(PRESETS).map((n) => (
          <button key={n} onClick={() => { setPreset(n); setWeights(presetWeights(n, assets)); }}
            className={`rounded px-2 py-0.5 text-xs ${preset === n ? "bg-panel-2 text-ink" : "text-muted hover:text-ink-2"}`}>
            {n}
          </button>
        ))}
        <span className="num ml-auto text-[11px] text-muted">{held.length} holdings</span>
        <button onClick={() => setOpen(!open)} className="text-xs text-amber hover:underline">{open ? "done" : "edit weights"}</button>
      </div>
      {open && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 border-t border-line px-4 py-3 sm:grid-cols-4 lg:grid-cols-6">
          {assets.map((a) => (
            <label key={a} className="flex items-center justify-between gap-2 text-xs">
              <span className="num text-ink-2">{a}</span>
              <span className="flex items-center gap-1">
                <input type="number" min={0} step={1} value={weights[a] ?? 0}
                  onChange={(e) => { setPreset("Custom"); setWeights({ ...weights, [a]: Math.max(0, Number(e.target.value)) }); }}
                  className="num w-14 rounded border border-line bg-bg px-1.5 py-0.5 text-right text-ink" />
                <span className="num w-10 text-right text-muted">{pct((weights[a] ?? 0) / total, 0)}</span>
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
