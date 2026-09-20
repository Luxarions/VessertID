/**
 * signals.js
 * Changes when: shutdown policy (which signals, timeout, force-exit) changes.
 * @memberof ve
 */

/**
 * @memberof ve
 * @param {(sig: string) => Promise<void>|void} onShutdown
 * @param {{forceAfterMs?: number, signals?: string[]}} [options]
 * @returns {() => void}
 */
function installSignals(onShutdown, options = {}) {
  const forceAfterMs = options.forceAfterMs ?? 10_000;
  const signals = options.signals ?? ['SIGTERM', 'SIGINT'];
  let running = false;
  const handlers = new Map();

  for (const sig of signals) {
    const handler = async () => {
      if (running) return;
      running = true;
      const timer = setTimeout(() => {
        process.stderr.write(`shutdown: force-exit after ${forceAfterMs}ms\n`);
        process.exit(1);
      }, forceAfterMs);
      timer.unref();
      try { await onShutdown(sig); }
      catch (e) { process.stderr.write(`shutdown: ${e.message}\n`); }
      finally { clearTimeout(timer); }
    };
    handlers.set(sig, handler);
    process.on(sig, handler);
  }

  return () => { for (const [s, h] of handlers) process.off(s, h); };
}

export { installSignals };
