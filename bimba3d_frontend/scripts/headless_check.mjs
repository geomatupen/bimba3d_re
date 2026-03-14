import puppeteer from 'puppeteer';
import fs from 'fs';

(async () => {
  const PROJECT_ID = '805cf5ea-f372-478d-9240-89b463fed6a8';
  const url = `http://localhost:5174/project/${PROJECT_ID}`;
  const outScreenshot = '/tmp/process_page.png';
  const outJson = '/tmp/process_page_inspect.json';

  const browser = await puppeteer.launch({ args: ['--no-sandbox','--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  const logs = [];
  page.on('console', msg => {
    try {
      logs.push({type: msg.type(), text: msg.text()});
    } catch (e) {
      logs.push({type: 'unknown', text: String(msg)});
    }
  });
  page.on('pageerror', err => logs.push({type: 'pageerror', text: err.toString()}));

  await page.setViewport({ width: 1280, height: 900 });
  await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 }).catch(e => { logs.push({type: 'error', text: 'goto failed: '+e.message}); });
  // wait a bit for SPA route and components to mount
  await new Promise((res) => setTimeout(res, 1200));

  // Ensure Process tab is active by clicking the tab if present
  try {
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const t = btns.find(b => /Process/i.test(b.textContent || ''));
      if (t) t.click();
    });
    await new Promise((res) => setTimeout(res, 800));
  } catch (e) { logs.push({type:'error', text:'tab click failed: '+e.message}); }

  // Inspect canvases and overlay elements
  const canvases = await page.evaluate(() => {
    const out = [];
    const nodes = Array.from(document.querySelectorAll('canvas'));
    nodes.forEach((c, i) => {
      const r = c.getBoundingClientRect();
      const style = window.getComputedStyle(c);
      let hasWebGL = false;
      try { hasWebGL = !!(c.getContext && (c.getContext('webgl') || c.getContext('webgl2'))); } catch(e) { hasWebGL = false; }
      out.push({index: i, width: c.width, height: c.height, clientW: c.clientWidth, clientH: c.clientHeight, rect: {x: r.x, y: r.y, w: r.width, h: r.height}, zIndex: style.zIndex, pointerEvents: style.pointerEvents, visibility: style.visibility, opacity: style.opacity, hasWebGL});
    });
    // Also inspect any absolutely positioned divs that might cover map
    const overlays = Array.from(document.querySelectorAll('div')).filter(d => {
      const s = window.getComputedStyle(d);
      return (s.position === 'absolute' || s.position === 'fixed') && (parseFloat(s.opacity||'1') > 0) && (d.offsetWidth>0 && d.offsetHeight>0);
    }).slice(0,50).map(d => { const s=window.getComputedStyle(d); const r=d.getBoundingClientRect(); return {tag:'div', class: d.className, rect:{x:r.x,y:r.y,w:r.width,h:r.height}, zIndex: s.zIndex, pointerEvents: s.pointerEvents}; });
    return {canvases: out, overlays: overlays};
  });
  // Inspect markers: look for elements with the marker HTML color used in app
  const markers = await page.evaluate(() => {
    const markers = Array.from(document.querySelectorAll('.maplibregl-marker'));
    return markers.slice(0, 200).map((m, i) => {
      const r = m.getBoundingClientRect();
      const s = window.getComputedStyle(m);
      return {
        index: i,
        class: m.className,
        html: (m.innerHTML || '').slice(0, 200),
        rect: {x: r.x, y: r.y, w: r.width, h: r.height},
        transform: s.transform,
        offsetParent: m.offsetParent ? m.offsetParent.tagName : null,
        pointerEvents: s.pointerEvents
      };
    });
  });

  await page.screenshot({ path: outScreenshot, fullPage: true }).catch(e => logs.push({type:'error', text:'screenshot failed: '+e.message}));

  const result = { url, timestamp: new Date().toISOString(), canvases, markers, logs };
  try { fs.writeFileSync(outJson, JSON.stringify(result, null, 2)); } catch (e) { console.error('write json failed', e); }
  console.log('INSPECT_JSON_PATH=' + outJson);
  console.log('SCREENSHOT=' + outScreenshot);
  await browser.close();
})();
