export default async function run(page, ui) {
  await page.waitForFunction(() => document.body.innerText.includes('Modelos y Ajustes'), null, { timeout: 30000 });
  await page.getByRole('tab', { name: 'Modelos y Ajustes' }).click();
  await page.waitForTimeout(1200);
  const snap = await ui.snapshot();
  const combo = snap.match(/@(e\d+) combobox "Modelo seleccionado"/)?.[1];
  await ui.click(combo);
  await page.waitForTimeout(500);
  await page.locator('[role=option]', { hasText: 'Animate-14B' }).first().click();
  await page.waitForTimeout(400);
  const snap2 = await ui.snapshot();
  const btn = snap2.match(/@(e\d+) button "⬇ Descargar"/)?.[1];
  await ui.click(btn);
  await page.waitForTimeout(20000);
  const bar = await page.evaluate(() => {
    const el = [...document.querySelectorAll('div')].find(d => /archivos ·|Sin descarga|completada/i.test(d.textContent) && d.textContent.length < 200);
    return el ? el.textContent.trim() : null;
  });
  const msg = await page.evaluate(() => {
    const el = [...document.querySelectorAll('div')].find(d => /Descarga iniciada|Error:/.test(d.textContent) && d.textContent.length < 160);
    return el?.textContent?.trim();
  });
  return { bar, msg };
}
