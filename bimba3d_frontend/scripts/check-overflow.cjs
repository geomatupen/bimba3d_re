const puppeteer = require('puppeteer');
(async () => {
  const url = process.argv[2] || 'http://localhost:5176';
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1366, height: 768 });
  await page.goto(url, { waitUntil: 'networkidle2' });
  await page.waitForTimeout(800);
  // run overflow detection
  const result = await page.evaluate(() => {
    const vw = document.documentElement.clientWidth;
    const els = Array.from(document.querySelectorAll('body *'));
    const overflowers = els.filter(el => {
      try { return el.getBoundingClientRect().width > vw + 1; } catch { return false; }
    }).map(el => {
      const rect = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      const path = (function getPath(e){ const parts=[]; while(e && e.nodeType===1 && e.tagName.toLowerCase()!=='html'){ let p=e.tagName.toLowerCase(); if(e.id) p += '#'+e.id; else if(e.className) p += '.'+Array.from(e.classList).slice(0,3).join('.'); parts.unshift(p); e=e.parentElement;} return parts.join(' > '); })(el);
      return {path, rect: {x:rect.x,y:rect.y,width:rect.width,height:rect.height}, computed: {width: cs.width, overflowX: cs.overflowX, boxSizing: cs.boxSizing}};
    });
    return {viewport: {width: vw, height: document.documentElement.clientHeight}, overflowers};
  });
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
})();
