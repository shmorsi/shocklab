"use client";
import { ReactNode } from "react";

export function Panel({ title, kicker, right, children, className = "" }: {
  title: string; kicker?: string; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`relative rounded-md border border-line bg-panel/90 backdrop-blur-sm ${className}`}>
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
        <div className="flex items-baseline gap-3">
          {kicker && <span className="num text-[10px] tracking-widest text-amber">{kicker}</span>}
          <h2 className="text-sm font-medium tracking-wide text-ink">{title}</h2>
        </div>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: string; color?: string }[]; onChange: (v: T) => void; label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex rounded border border-line p-0.5 text-xs">
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}
          className={`flex items-center gap-1.5 rounded-sm px-2.5 py-1 transition-colors ${value === o.value ? "bg-panel-2 text-ink" : "text-muted hover:text-ink-2"}`}>
          {o.color && <span className="h-2 w-2 rounded-full" style={{ background: o.color }} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Small floating tooltip; parent must be position:relative. */
export function Tip({ x, y, children }: { x: number; y: number; children: ReactNode }) {
  return (
    <div className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-full whitespace-nowrap rounded border border-line bg-bg/95 px-2 py-1 text-[11px] text-ink shadow-lg"
      style={{ left: x, top: y - 8 }}>
      {children}
    </div>
  );
}

export const MODEL_OPTS = [
  { value: "mdn" as const, label: "MDN", color: "var(--mdn)" },
  { value: "baseline" as const, label: "Baseline", color: "var(--base)" },
];
