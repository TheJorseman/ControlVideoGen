export default async function run(page, ui) {
  await page.waitForFunction(() => document.body.innerText.includes('Modelos y Ajustes'), null, { timeout: 30000 });
  await page.getByRole('tab', { name: 'Modelos y Ajustes' }).click();
  await page.waitForTimeout(1000);

  const snap = await ui.snapshot();
  const combo = snap.match(/@(e\d+) combobox "Modelo seleccionado"/)?.[1];
  if (!combo) return { error: 'dropdown no encontrado', snap: snap.slice(0, 600) };
  await ui.click(combo);
  await page.waitForTimeout(500);
  await page.locator('[role=option]', { hasText: 'Animate-14B' }).first().click();
  await page.waitForTimeout(400);

  const snap2 = await ui.snapshot();
  const btn = snap2.match(/@(e\d+) button "⬇ Descargar"/)?.[1];
  const barRead = () => page.evaluate(() => {
    const el = [...document.querySelectorAll('div')].find(d => /archivos ·|Sin descarga|completada/i.test(d.textContent) && d.textContent.length < 200);
    return el ? el.textContent.trim() : null;
  });

  const barBefore = await barRead();
  await ui.click(btn);
  await page.waitForTimeout(30000);
  const bar1 = await barRead();
  await page.waitForTimeout(45000);
  const bar2 = await barRead();
  const msg = await page.evaluate(() => {
    const el = [...document.querySelectorAll('div')].find(d => /Descarga iniciada|Error:/.test(d.textContent) && d.textContent.length < 160);
    return el?.textContent?.trim();
  });
  return { barBefore, bar1, bar2, advanced: bar1 !== bar2 && bar2 !== null, msg };
}
