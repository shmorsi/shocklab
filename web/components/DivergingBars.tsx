"use client";
import { motion } from "framer-motion";

export type Row = { label: string; value: number; note?: string };

/** Horizontal bars around a centre line. `lossIsPositive`: for risk numbers (CVaR
 * contribution) a positive value is a loss and paints orange; for P&L it's the reverse. */
export default function DivergingBars({ rows, fmt, lossIsPositive = false, selected, onSelect, maxRows }: {
  rows: Row[]; fmt: (x: number) => string; lossIsPositive?: boolean;
  selected?: string | null; onSelect?: (label: string) => void; maxRows?: number;
}) {
  const shown = maxRows ? rows.slice(0, maxRows) : rows;
  const max = Math.max(1e-9, ...shown.map((r) => Math.abs(r.value)));
  return (
    <ul className="space-y-[3px]">
      {shown.map((r) => {
        const isLoss = lossIsPositive ? r.value > 0 : r.value < 0;
        const w = (Math.abs(r.value) / max) * 50;
        const color = isLoss ? "var(--loss)" : "var(--gain)";
        const active = selected === r.label;
        return (
          <motion.li layout key={r.label} transition={{ type: "spring", stiffness: 260, damping: 30 }}
            onClick={onSelect ? () => onSelect(r.label) : undefined}
            className={`group grid grid-cols-[minmax(68px,96px)_1fr_64px] items-center gap-2 rounded-sm px-1 py-[3px] text-xs ${onSelect ? "cursor-pointer hover:bg-panel-2" : ""} ${active ? "bg-panel-2 ring-1 ring-amber/50" : ""}`}
            title={r.note}>
            <span className={`num truncate ${active ? "text-amber" : "text-ink-2"}`}>{r.label}</span>
            <span className="relative h-3">
              <span className="absolute left-1/2 top-[-3px] h-[18px] w-px bg-line" />
              <motion.span initial={false} className="absolute top-0 h-3 rounded-[2px]"
                animate={{ width: `${w}%`, left: r.value < 0 ? `${50 - w}%` : "50%" }}
                transition={{ type: "spring", stiffness: 200, damping: 26 }}
                style={{ background: color, opacity: 0.85 }} />
            </span>
            <span className="num text-right text-ink">{fmt(r.value)}</span>
          </motion.li>
        );
      })}
    </ul>
  );
}
