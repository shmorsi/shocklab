"use client";
import type { EventReplay, Regime, Shock } from "@/lib/api";
import { Panel } from "./ui";

export const ZERO_SHOCK: Shock = { oil_pct: 0, rates_bp: 0, usd_pct: 0, vix_pct: 0, credit_bp: 0 };

const SLIDERS: { key: keyof Shock; label: string; min: number; max: number; step: number; unit: "%" | "bp" }[] = [
  { key: "oil_pct", label: "WTI oil", min: -60, max: 60, step: 1, unit: "%" },
  { key: "rates_bp", label: "US 10Y yield", min: -200, max: 300, step: 5, unit: "bp" },
  { key: "usd_pct", label: "Broad USD", min: -10, max: 10, step: 0.5, unit: "%" },
  { key: "vix_pct", label: "VIX", min: -50, max: 200, step: 5, unit: "%" },
  { key: "credit_bp", label: "Credit spread (Baa−10Y)", min: -150, max: 400, step: 5, unit: "bp" },
];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function clampShock(s: Shock): Shock {
  const out = { ...s };
  for (const sl of SLIDERS) out[sl.key] = Math.round(clamp(s[sl.key], sl.min, sl.max) / sl.step) * sl.step;
  return out;
}

export default function ShockPanel({ shock, setShock, events, activePreset, onPreset, regime, regimeLabel, onResetRegime }: {
  shock: Shock;
  setShock: (s: Shock) => void;
  events: EventReplay[];
  activePreset: string | null;
  onPreset: (e: EventReplay | null) => void;
  regime: Regime | null;
  regimeLabel: string;
  onResetRegime: () => void;
}) {
  return (
    <Panel kicker="01" title="Macro shock" right={<span className="num text-[11px] text-muted">10-trading-day move</span>}>
      <div className="space-y-3.5">
        {SLIDERS.map((s) => {
          const v = shock[s.key];
          const zeroPos = ((0 - s.min) / (s.max - s.min)) * 100;
          const pos = ((v - s.min) / (s.max - s.min)) * 100;
          return (
            <label key={s.key} className="block">
              <div className="flex items-baseline justify-between text-xs">
                <span className="text-ink-2">{s.label}</span>
                <span className={`num text-sm ${v === 0 ? "text-muted" : "text-amber"}`}>
                  {v > 0 ? "+" : v < 0 ? "−" : ""}{Math.abs(v).toFixed(s.step < 1 ? 1 : 0)}{s.unit}
                </span>
              </div>
              <div className="relative">
                {/* amber fill from zero to the thumb shows the size of the shock */}
                <div className="pointer-events-none absolute top-[10px] h-[2px] bg-amber/70"
                  style={{ left: `${Math.min(zeroPos, pos)}%`, width: `${Math.abs(pos - zeroPos)}%` }} />
                <div className="pointer-events-none absolute top-[6px] h-[10px] w-px bg-muted" style={{ left: `${zeroPos}%` }} />
                <input type="range" min={s.min} max={s.max} step={s.step} value={v} aria-label={s.label}
                  onChange={(e) => setShock({ ...shock, [s.key]: Number(e.target.value) })} />
              </div>
            </label>
          );
        })}
      </div>
      <div className="mt-4 border-t border-line pt-3">
        <div className="mb-2 text-[11px] uppercase tracking-widest text-muted">Replay a historical shock</div>
        <div className="flex flex-wrap gap-1.5">
          {events.map((e) => (
            <button key={e.name} onClick={() => onPreset(e)}
              className={`rounded border px-2 py-1 text-xs transition-colors ${activePreset === e.name ? "border-amber bg-amber/10 text-amber" : "border-line text-ink-2 hover:border-amber/60 hover:text-ink"}`}>
              {e.name}
            </button>
          ))}
          <button onClick={() => onPreset(null)} className="rounded border border-line px-2 py-1 text-xs text-muted hover:text-ink">
            Reset
          </button>
        </div>
        {regime && (
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
            <span>Regime: <span className="text-ink-2">{regimeLabel}</span></span>
            <span className="num">VIX {regime.vix_level.toFixed(1)} · avg corr {regime.avg_corr.toFixed(2)}</span>
            {regimeLabel !== "today" && (
              <button onClick={onResetRegime} className="underline decoration-dotted hover:text-ink">use today&apos;s</button>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
