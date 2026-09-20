export default async function run(page) {
  await page.waitForFunction(() => document.body.innerText.includes('Modelos y Ajustes'), null, { timeout: 30000 });
  await page.getByRole('tab', { name: 'Modelos y Ajustes' }).click();
  await page.waitForTimeout(3000);
  const bar = await page.evaluate(() => {
    const el = [...document.querySelectorAll('div')].find(d => /archivos ·|Sin descarga|completada|Error/i.test(d.textContent) && d.textContent.length < 250);
    return el ? el.textContent.trim() : null;
  });
  return { bar };
}
