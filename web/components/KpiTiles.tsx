"use client";
import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { useEffect } from "react";

import type { ModelStress } from "@/lib/api";
import { pct, spct } from "@/lib/format";

/** Number that tweens to its new value instead of jumping. */
function Tween({ value, fmt }: { value: number; fmt: (x: number) => string }) {
  const mv = useMotionValue(value);
  const text = useTransform(mv, fmt);
  useEffect(() => {
    const c = animate(mv, value, { duration: 0.45, ease: "easeOut" });
    return c.stop;
  }, [value, mv]);
  return <motion.span>{text}</motion.span>;
}

export default function KpiTiles({ m, other, otherLabel }: { m: ModelStress; other: ModelStress; otherLabel: string }) {
  const drag = m.contributions.find((c) => c.asset === m.biggest_drag);
  const tiles = [
    { label: "Expected P&L", value: m.expected, fmt: (x: number) => spct(x, 2), tone: m.expected < 0 ? "text-loss" : "text-gain", alt: other.expected },
    { label: "VaR 95", value: m.var95, fmt: (x: number) => pct(x, 2), tone: "text-ink", alt: other.var95, sub: "loss not exceeded 19 of 20 times" },
    { label: "CVaR 95", value: m.cvar95, fmt: (x: number) => pct(x, 2), tone: "text-ink", alt: other.cvar95, sub: "average loss in the worst 5%" },
  ];
  return (
    <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
      {tiles.map((t) => (
        <div key={t.label} className="rounded-md border border-line bg-panel/90 px-4 py-3">
          <div className="text-[11px] uppercase tracking-widest text-muted">{t.label}</div>
          <div className={`num mt-1 text-2xl font-medium sm:text-3xl ${t.tone}`}><Tween value={t.value} fmt={t.fmt} /></div>
          <div className="num mt-1 text-[11px] text-muted">{otherLabel} {t.fmt(t.alt)}</div>
        </div>
      ))}
      <div className="rounded-md border border-line bg-panel/90 px-4 py-3">
        <div className="text-[11px] uppercase tracking-widest text-muted">Biggest drag</div>
        <div className="num mt-1 text-2xl font-medium text-ink sm:text-3xl">{m.biggest_drag}</div>
        <div className="num mt-1 text-[11px] text-muted">
          {drag ? `${pct(drag.cvar_contrib / m.cvar95, 0)} of CVaR at ${pct(drag.weight, 1)} weight` : ""}
        </div>
      </div>
    </div>
  );
}
