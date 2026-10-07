"use client";
import * as d3 from "d3";
import { motion } from "framer-motion";
import { useState } from "react";

import { pct, spct } from "@/lib/format";
import { useWidth } from "@/lib/hooks";
import { Tip } from "./ui";

type Pt = { cvar: number; exp_ret: number };

/** Efficient frontier: stressed CVaR (x) vs unconditional expected return (y). */
export default function Frontier({ points, before, after }: { points: Pt[]; before: Pt; after: Pt }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ p: Pt; label: string } | null>(null);
  const h = 220, m = { t: 12, r: 12, b: 30, l: 46 };
  const w = Math.max(width, 220);
  const all = [...points, before, after];
  const x = d3.scaleLinear().domain(d3.extent(all, (d) => d.cvar) as [number, number]).nice().range([m.l, w - m.r]);
  const y = d3.scaleLinear().domain(d3.extent(all, (d) => d.exp_ret) as [number, number]).nice().range([h - m.b, m.t]);
  const line = d3.line<Pt>().x((d) => x(d.cvar)).y((d) => y(d.exp_ret)).curve(d3.curveMonotoneX);
  const marks = [
    { p: before, label: "Current", fill: "var(--bg)", stroke: "var(--ink-2)" },
    { p: after, label: "Suggested", fill: "var(--amber)", stroke: "var(--bg)" },
  ];
  return (
    <div ref={ref} className="relative w-full" style={{ height: h }}>
      {width > 0 && (
        <svg width={w} height={h} role="img" aria-label="CVaR versus expected return frontier">
          {x.ticks(4).map((t) => (
            <g key={`x${t}`}>
              <line x1={x(t)} x2={x(t)} y1={m.t} y2={h - m.b} stroke="var(--grid)" />
              <text x={x(t)} y={h - 12} textAnchor="middle" fontSize={10} className="num" fill="var(--muted)">{pct(t, 1)}</text>
            </g>
          ))}
          {y.ticks(4).map((t) => (
            <g key={`y${t}`}>
              <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
              <text x={m.l - 6} y={y(t) + 3} textAnchor="end" fontSize={10} className="num" fill="var(--muted)">{pct(t, 2)}</text>
            </g>
          ))}
          <text x={w - m.r} y={h - 1} textAnchor="end" fontSize={10} fill="var(--muted)">stressed CVaR 95 →</text>
          <motion.path initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.8 }}
            d={line(points) ?? ""} fill="none" stroke="var(--ink-2)" strokeWidth={2} />
          {points.map((p, i) => (
            <circle key={i} cx={x(p.cvar)} cy={y(p.exp_ret)} r={2.5} fill="var(--ink-2)" />
          ))}
          {marks.map((mk) => (
            <g key={mk.label}>
              <circle cx={x(mk.p.cvar)} cy={y(mk.p.exp_ret)} r={5} fill={mk.fill} stroke={mk.stroke} strokeWidth={2} />
              <text x={x(mk.p.cvar) + (x(mk.p.cvar) > w - 80 ? -9 : 9)} y={y(mk.p.exp_ret) + 3}
                textAnchor={x(mk.p.cvar) > w - 80 ? "end" : "start"} fontSize={10} fill="var(--ink)">{mk.label}</text>
              <circle cx={x(mk.p.cvar)} cy={y(mk.p.exp_ret)} r={14} fill="transparent"
                onMouseEnter={() => setHover(mk)} onMouseLeave={() => setHover(null)} />
            </g>
          ))}
          {points.map((p, i) => (
            <circle key={`h${i}`} cx={x(p.cvar)} cy={y(p.exp_ret)} r={9} fill="transparent"
              onMouseEnter={() => setHover({ p, label: "Frontier" })} onMouseLeave={() => setHover(null)} />
          ))}
        </svg>
      )}
      {hover && (
        <Tip x={x(hover.p.cvar)} y={y(hover.p.exp_ret)}>
          <span className="text-ink-2">{hover.label}</span>{" "}
          <span className="num">CVaR {pct(hover.p.cvar, 2)} · E[r] {spct(hover.p.exp_ret, 2)}</span>
        </Tip>
      )}
    </div>
  );
}
