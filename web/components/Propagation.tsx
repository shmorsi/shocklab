"use client";
import { useState } from "react";

import type { ModelStress } from "@/lib/api";
import { FACTOR_LABEL, spct } from "@/lib/format";
import DivergingBars from "./DivergingBars";
import { Panel } from "./ui";

const FACTORS = ["oil", "rates", "usd", "vix", "credit"];

/** Factor -> asset: left, how much each factor moves the portfolio's expected return;
 * right, which holdings carry that factor's hit (click a factor to filter). */
export default function Propagation({ m, model }: { m: ModelStress; model: string }) {
  const [factor, setFactor] = useState<string | null>(null);
  const label = (f: string) => FACTOR_LABEL[f] ?? f;
  const factorRows = m.factor_attribution
    .filter((f) => !(model === "baseline" && f.factor === "interaction"))
    .map((f) => ({ label: label(f.factor), value: f.contribution, key: f.factor }));
  const key = factor ? factorRows.find((r) => r.label === factor)?.key : null;
  const assetRows = m.asset_factor
    .map((a) => ({
      label: a.asset,
      value: key && FACTORS.includes(key) ? (a[key] as number) : FACTORS.reduce((s, f) => s + (a[f] as number), 0),
    }))
    .sort((a, b) => a.value - b.value);
  const nonFactor = key === "carry" || key === "interaction";
  return (
    <Panel kicker="04" title="Shock propagation" right={<span className="text-[11px] text-muted">contribution to expected 10-day return</span>}>
      <div className="grid gap-5 md:grid-cols-2">
        <div>
          <div className="mb-2 text-[11px] uppercase tracking-widest text-muted">Factor → portfolio</div>
          <DivergingBars rows={factorRows} fmt={(x) => spct(x, 2)} selected={factor}
            onSelect={(l) => setFactor(factor === l ? null : l)} />
          <p className="mt-3 text-[11px] leading-relaxed text-muted">
            {model === "baseline"
              ? "Linear model: each bar is exactly Σ weight × beta × shock. Carry is the intercept."
              : "MDN: each bar moves one factor alone vs. no shock. Interaction is what the network does when factors move together."}
            {" "}Click a factor to see which holdings carry it.
          </p>
        </div>
        <div>
          <div className="mb-2 text-[11px] uppercase tracking-widest text-muted">
            {factor ? `${factor} → holdings` : "All factors → holdings"}
          </div>
          {nonFactor ? (
            <p className="text-xs text-muted">Not split by holding.</p>
          ) : (
            <DivergingBars rows={assetRows} fmt={(x) => spct(x, 2)} maxRows={12} />
          )}
        </div>
      </div>
    </Panel>
  );
}
