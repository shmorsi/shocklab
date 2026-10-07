"use client";
import * as d3 from "d3";
import { useState } from "react";

import { useWidth } from "@/lib/hooks";
import { pct, spct } from "@/lib/format";
import { Tip } from "./ui";

// Bars carry their final geometry as attributes (always correct, even without JS
// animation); CSS transitions morph them when a new distribution arrives.
const MORPH: React.CSSProperties = {
  transition: "x 380ms cubic-bezier(.2,.8,.2,1), y 380ms cubic-bezier(.2,.8,.2,1), width 380ms, height 380ms cubic-bezier(.2,.8,.2,1), transform 380ms cubic-bezier(.2,.8,.2,1), opacity 150ms",
};

export type VLine = { x: number; label: string; dash?: boolean };

/** P&L distribution: filled bars (loss = orange, gain = blue) that morph between
 * responses, an optional ghost outline (e.g. "before"), and VaR/CVaR markers. */
export default function Histogram({ edges, density, ghost, lines = [], height = 240 }: {
  edges: number[]; density: number[]; ghost?: number[]; lines?: VLine[]; height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const m = { t: 34, r: 8, b: 26, l: 8 };
  const w = Math.max(width, 200);
  const x = d3.scaleLinear().domain([edges[0], edges[edges.length - 1]]).range([m.l, w - m.r]);
  const ymax = d3.max([...density, ...(ghost ?? [])]) ?? 1;
  const y = d3.scaleLinear().domain([0, ymax * 1.1]).range([height - m.b, m.t]);
  const bw = Math.max(1, x(edges[1]) - x(edges[0]) - 1.5);
  const ticks = x.ticks(w < 420 ? 4 : 7);
  const ghostPath = ghost
    ? d3.line<number>().x((_, i) => x(edges[i])).y((d) => y(d)).curve(d3.curveStepAfter)([...ghost, ghost[ghost.length - 1]])
    : null;

  return (
    <div ref={ref} className="relative w-full" style={{ height }}>
      {width > 0 && (
        <svg width={w} height={height} role="img" aria-label="Distribution of simulated 10-day portfolio returns">
          {ticks.map((t) => (
            <g key={t} transform={`translate(${x(t)},0)`}>
              <line y1={m.t} y2={height - m.b} stroke="var(--grid)" />
              <text y={height - 8} textAnchor="middle" className="num" fontSize={10} fill="var(--muted)">{pct(t, 0)}</text>
            </g>
          ))}
          {density.map((d, i) => {
            const mid = (edges[i] + edges[i + 1]) / 2;
            return (
              <rect key={i} rx={1.5} x={x(edges[i]) + 0.75} width={bw} y={y(d)} height={Math.max(0, y(0) - y(d))}
                style={MORPH}
                fill={mid < 0 ? "var(--loss)" : "var(--gain)"} opacity={hover === null || hover === i ? 0.9 : 0.45}
              />
            );
          })}
          {ghostPath && <path d={ghostPath} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="3 3" />}
          <line x1={m.l} x2={w - m.r} y1={y(0)} y2={y(0)} stroke="var(--line)" />
          {lines.map((l, k) => (
            <g key={l.label} style={{ transform: `translateX(${x(l.x)}px)`, ...MORPH }}>
              <line y1={m.t - 4} y2={height - m.b} stroke="var(--ink)" strokeWidth={1.25} strokeDasharray={l.dash ? "4 3" : undefined} />
              {/* one label row per marker; anchor away from the nearest edge so text never clips */}
              <text y={10 + k * 12} x={x(l.x) > w - 90 ? -4 : 4} textAnchor={x(l.x) > w - 90 ? "end" : "start"} className="num" fontSize={10} fill="var(--ink)">
                {l.label} {pct(-l.x)}
              </text>
            </g>
          ))}
          {/* invisible full-height hit targets, wider than the bars */}
          {density.map((_, i) => (
            <rect key={i} x={x(edges[i])} width={x(edges[i + 1]) - x(edges[i])} y={m.t} height={height - m.t - m.b}
              fill="transparent" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
          ))}
        </svg>
      )}
      {hover !== null && width > 0 && (
        <Tip x={x((edges[hover] + edges[hover + 1]) / 2)} y={y(density[hover])}>
          <span className="num">{spct(edges[hover])} to {spct(edges[hover + 1])}</span>
          <span className="ml-2 text-ink-2">p = <span className="num text-ink">{pct(density[hover], 2)}</span></span>
          {ghost && <span className="ml-2 text-ink-2">before <span className="num">{pct(ghost[hover], 2)}</span></span>}
        </Tip>
      )}
    </div>
  );
}
