export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const spct = (x: number, d = 1) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(d)}%`;
export const bp = (x: number) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(0)}bp`;

export const FACTOR_LABEL: Record<string, string> = {
  oil: "Oil",
  rates: "10Y yield",
  usd: "USD",
  vix: "VIX",
  credit: "Credit spread",
  carry: "Carry",
  interaction: "Interaction",
};
