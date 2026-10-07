// Capture README screenshots with headless Chrome over the DevTools protocol.
// Needs the API (:8000) and web (:3000) running. Usage: node docs/screenshots.mjs
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";

const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = process.env.URL ?? "http://localhost:3000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const proc = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9333", "--hide-scrollbars", "--user-data-dir=/tmp/shocklab-shots", "about:blank"], { stdio: "ignore" });
await sleep(2000);
const targets = await (await fetch("http://127.0.0.1:9333/json")).json();
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
ws.addEventListener("message", (e) => {
  const msg = JSON.parse(e.data);
  if (pending.has(msg.id)) { pending.get(msg.id)(msg.result); pending.delete(msg.id); }
});
const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }))?.result?.value;

async function shot(file, { width, height, mobile = false, section = null, full = false }) {
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 2, mobile });
  await send("Page.navigate", { url: URL });
  await sleep(7000); // stress call + one optimiser solve + entry animations
  let clip = null;
  if (section) {
    const r = await evaluate(`(() => { const s=[...document.querySelectorAll('section')].find(e=>e.querySelector('h2')?.textContent==='${section}'); const b=s.getBoundingClientRect(); return {x:b.x-8,y:b.y+scrollY-8,width:b.width+16,height:b.height+16}; })()`);
    clip = { ...r, scale: 1 };
  } else if (!full) {
    clip = { x: 0, y: 0, width, height, scale: 1 };
  } else {
    const h = await evaluate("document.documentElement.scrollHeight");
    clip = { x: 0, y: 0, width, height: h, scale: 1 };
  }
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, ...(clip ? { clip } : {}) });
  writeFileSync(file, Buffer.from(data, "base64"));
  console.log("wrote", file);
}

await send("Page.enable");
const out = process.env.OUT ?? "docs/screenshots";
await shot(`${out}/hero.png`, { width: 1440, height: 900 });
await shot(`${out}/desktop-full.png`, { width: 1440, height: 900, full: true });
await shot(`${out}/optimizer.png`, { width: 1440, height: 900, section: "CVaR optimiser" });
await shot(`${out}/replay.png`, { width: 1440, height: 900, section: "Backtest replay" });
await shot(`${out}/honesty.png`, { width: 1440, height: 900, section: "Model honesty" });
await shot(`${out}/mobile.png`, { width: 390, height: 844, mobile: true });
ws.close();
proc.kill();
