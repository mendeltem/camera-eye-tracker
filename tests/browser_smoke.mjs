// End-to-end check in headless Chromium with a fake camera that plays a face clip.
//   FAKE_VIDEO=face.y4m node tests/browser_smoke.mjs        (needs: npm i playwright && npx playwright install chromium)
// A still or slowly drifting photo cannot follow the dots, so this checks the pipeline, not the accuracy:
// camera -> face model -> setup checks -> calibration -> model fit -> validation -> tracking views, no page errors.
import { createRequire } from 'node:module';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const VIDEO = process.env.FAKE_VIDEO;
const OUT = process.env.SHOTS_DIR || '';
if (!VIDEO || !fs.existsSync(VIDEO)) { console.error('set FAKE_VIDEO to a .y4m face clip'); process.exit(2); }

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = http.createServer((req, res) => {
  const file = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/\/$/, '/index.html'));
  if (!file.startsWith(ROOT) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
}).listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
const url = `http://127.0.0.1:${server.address().port}/?test=1`;

const proxy = process.env.https_proxy || process.env.HTTPS_PROXY;
const browser = await chromium.launch({
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${VIDEO}`],
  ...(proxy ? { proxy: { server: proxy, bypass: '127.0.0.1,localhost' } } : {}),
});
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
const problems = [];
page.on('pageerror', e => problems.push(`page error: ${e.message}`));
// MediaPipe prints its own INFO lines through console.error; only real errors count
page.on('console', m => { if (m.type() === 'error' && !/^INFO:/.test(m.text())) problems.push(`console: ${m.text()}`); });
const shot = async name => { if (OUT) await page.screenshot({ path: path.join(OUT, `${name}.png`) }); };
const results = [];
const step = async (name, fn) => {
  try { const info = await fn(); results.push(`ok    ${name}${info ? `  (${info})` : ''}`); }
  catch (e) { results.push(`FAIL  ${name}: ${e.message.split('\n')[0]}`); throw e; }
};

try {
  await page.goto(url);
  await step('intro screen shows', () => page.waitForSelector('#btnStart:not([disabled])'));
  await shot('1-intro');
  await page.click('#btnStart');
  await step('camera and face model load', async () => {
    await page.waitForSelector('body[data-mode="setup"]', { timeout: 90000 });
    return await page.evaluate(() => `delegate ${window.gazeApp.delegate}`);
  });
  await step('face and irises found (drawn without the camera picture)', async () => {
    await page.waitForSelector('#btnCalibrate:not([disabled])', { timeout: 30000 });
    await page.waitForTimeout(1500);
    return await page.$$eval('.checks li', ls => ls.map(l => `${l.dataset.check}=${l.dataset.state}`).join(' '));
  });
  await shot('2-setup');
  await page.click('#btnCalibrate');
  await page.waitForTimeout(1500);
  await shot('3-calibration');
  await step('calibration and test dots finish', async () => {
    await page.waitForSelector('body[data-mode="track"]', { timeout: 120000 });
    return await page.evaluate(() => {
      const s = window.gazeApp;
      return `${s.train.length} samples, lambda ${s.model.lambda}, test error ${s.val ? Math.round(s.val.meanPx) + ' px' : 'none'}`;
    });
  });
  await step('result card shows the measured error', async () => (await page.textContent('#resultText')).slice(0, 70));
  await shot('4-result');
  await page.click('#btnOk');
  await step('gaze dot moves on screen', async () => {
    await page.waitForTimeout(1500);
    const g = await page.evaluate(() => window.gazeApp.gaze);
    if (!g || !Number.isFinite(g.x) || !Number.isFinite(g.y)) throw new Error('no gaze estimate');
    return `x ${g.x.toFixed(2)}, y ${g.y.toFixed(2)}`;
  });
  await step('tap correction refits the model', async () => {
    const before = await page.evaluate(() => window.gazeApp.train.length);
    await page.mouse.click(200, 500);
    await page.waitForTimeout(800);
    const after = await page.evaluate(() => window.gazeApp.train.length);
    if (after <= before) throw new Error(`training set ${before} -> ${after}`);
    return `${before} -> ${after} samples`;
  });
  await step('figures view: three figures, at most one lit', async () => {
    await page.waitForTimeout(1500);
    const n = await page.$$eval('.fig', f => f.length), lit = await page.$$eval('.fig.look', f => f.map(x => x.dataset.name));
    if (n !== 3 || lit.length > 1) throw new Error(`${n} figures, lit: ${lit}`);
    return `lit: ${lit[0] || 'none'}`;
  });
  await step('camera picture is not shown', async () => {
    const box = await page.$eval('#video', v => v.getBoundingClientRect().width);
    if (box > 4) throw new Error(`video is ${box}px wide`);
  });
  await shot('5-figures');
  await page.click('.seg button[data-view="dot"]');
  await page.waitForTimeout(500);
  await page.click('.seg button[data-view="heat"]');
  await page.waitForTimeout(1500);
  await shot('6-heatmap');
  await step('no page errors', async () => { if (problems.length) throw new Error(problems.join(' | ')); });
} catch (e) { if (!results.some(r => r.startsWith('FAIL'))) results.push(`FAIL  ${e.message.split('\n')[0]}`); } finally {
  console.log(results.join('\n'));
  if (problems.length) console.log('problems:\n  ' + problems.join('\n  '));
  await browser.close();
  server.close();
}
process.exit(results.some(r => r.startsWith('FAIL')) || results.length < 10 ? 1 : 0);
