"use client";
import type { ModelStress } from "@/lib/api";
import { pct } from "@/lib/format";
import DivergingBars from "./DivergingBars";
import { Panel } from "./ui";

/** Component CVaR: each holding's average loss inside the portfolio's worst 5%.
 * The bars add up to CVaR; negative bars are holdings that hedge the tail. */
export default function RiskAttribution({ m }: { m: ModelStress }) {
  const rows = m.contributions
    .map((c) => ({ label: c.asset, value: c.cvar_contrib, note: `weight ${pct(c.weight, 1)}` }))
    .sort((a, b) => b.value - a.value);
  return (
    <Panel kicker="05" title="Risk attribution by holding"
      right={<span className="num text-[11px] text-muted">Σ = CVaR {pct(m.cvar95, 2)}</span>}>
      <DivergingBars rows={rows} fmt={(x) => pct(x, 2)} lossIsPositive />
      <p className="mt-3 text-[11px] text-muted">
        <span className="text-loss">■</span> adds to tail loss · <span className="text-gain">■</span> hedges it. Each bar is the holding&apos;s average
        loss in the scenarios where the portfolio is in its worst 5%.
      </p>
    </Panel>
  );
}
