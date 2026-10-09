/*
 * How every test of the page is run. A test file ends with
 *
 *     require('./harness').run(__filename, name => eval(name));
 *
 * and every function in it whose name starts with test_ — plain or async — is
 * run, in order of name, awaited, and shown as a dot or an F, then a summary;
 * the process ends with 1 if anything failed. The callback looks a name up in
 * the test file's own scope, which is what lets the harness find the tests
 * itself: each file used to list them by hand, beside a check that the list
 * still matched what was written.
 */
const fs = require('fs');

async function run(file, find) {
  const names = [...fs.readFileSync(file, 'utf8').matchAll(/^(?:async )?function (test_\w+)/gm)]
    .map(match => match[1])
    .sort((a, b) => a.localeCompare(b));
  const failures = [];
  const started = Date.now();
  for (const name of names) {
    try {
      // Awaited, so an async test that rejects is a failure rather than a
      // pass with an unhandled rejection printed somewhere after the summary.
      await find(name)();
      process.stdout.write('.');
    } catch (err) {
      failures.push([name, err]);
      process.stdout.write('F');
    }
  }
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);
  console.log(`\n\n${names.length - failures.length} passed, ${failures.length} failed `
            + `in ${elapsed}s`);
  for (const [name, err] of failures) console.log(`\n--- ${name} ---\n${err.stack}`);
  process.exit(failures.length ? 1 : 0);
}

module.exports = { run };
