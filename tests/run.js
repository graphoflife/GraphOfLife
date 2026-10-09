/*
 * Every test of the page, one file after another:
 *
 *     node tests/run.js
 *
 * Each file runs in a process of its own, because each sets up the globals a
 * browser would have and the scripts it tests expect them to be theirs alone.
 */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const failed = [];
for (const name of fs.readdirSync(__dirname).filter(n => /^test_.*\.js$/.test(n)).sort()) {
  console.log(`\n${name}`);
  const { status } = spawnSync(process.execPath, [path.join(__dirname, name)], { stdio: 'inherit' });
  if (status !== 0) failed.push(name);
}
console.log(failed.length ? `\nfailed: ${failed.join(', ')}` : '\nevery file passed');
process.exit(failed.length ? 1 : 0);
