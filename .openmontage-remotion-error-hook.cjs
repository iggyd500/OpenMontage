function describe(value, depth = 0, seen = new Set()) {
  const pad = ' '.repeat(depth * 2);
  if (value === null || value === undefined) { console.error(pad + String(value)); return; }
  if ((typeof value === 'object' || typeof value === 'function') && seen.has(value)) { console.error(pad + '[circular]'); return; }
  if (typeof value === 'object' || typeof value === 'function') seen.add(value);
  console.error(pad + 'type=' + typeof value + ' constructor=' + (value?.constructor?.name || 'unknown'));
  const keys = new Set(); let cursor = value;
  while (cursor && cursor !== Object.prototype && keys.size < 100) { for (const key of Object.getOwnPropertyNames(cursor)) keys.add(key); cursor = Object.getPrototypeOf(cursor); }
  for (const key of keys) {
    if (['constructor','__defineGetter__','__defineSetter__','__lookupGetter__','__lookupSetter__','hasOwnProperty','isPrototypeOf','propertyIsEnumerable','toString','valueOf','toLocaleString'].includes(key)) continue;
    let item; try { item = value[key]; } catch(e) { item = '[getter threw: ' + e + ']'; }
    if (item === undefined || typeof item === 'function') continue;
    if (item && typeof item === 'object' && depth < 3) { console.error(pad + key + ':'); describe(item, depth + 1, seen); }
    else console.error(pad + key + '=' + (typeof item === 'string' ? item : JSON.stringify(item)));
  }
  if (value.stack) console.error(pad + 'STACK\n' + value.stack);
}
process.on('unhandledRejection', (reason) => { console.error('=== CAPTURED UNHANDLED REJECTION ==='); describe(reason); console.error('=== END CAPTURE ==='); process.exitCode = 1; });
process.on('uncaughtExceptionMonitor', (error, origin) => { console.error('=== UNCAUGHT EXCEPTION MONITOR origin=' + origin + ' ==='); describe(error); console.error('=== END MONITOR ==='); });
