/**
 * middleware.js
 * Changes when: the middleware composition model changes.
 * @memberof ve
 */

/**
 * Compose middlewares into a single (req, res) => Promise<void> runner.
 * Settles when the response finishes/closes or when `next(err)` is called.
 * @memberof ve
 * @param {Function[]} middlewares
 * @returns {Function}
 */
function compose(middlewares) {
  return function run(req, res) {
    return new Promise((resolve, reject) => {
      let idx = 0, settled = false;
      const done = (err) => { if (settled) return; settled = true; err ? reject(err) : resolve(); };
      res.once('finish', () => done());
      res.once('close', () => done());
      const next = (err) => {
        if (err) return done(err);
        if (idx >= middlewares.length) return done();
        const mw = middlewares[idx++];
        try {
          const result = mw(req, res, next);
          if (result && typeof result.then === 'function') result.catch(done);
        } catch (e) { done(e); }
      };
      next();
    });
  };
}

/**
 * @memberof ve
 * @param {(req:*, res:*) => Promise<void>|void} handler
 * @returns {Function}
 */
function asMiddleware(handler) {
  return async function wrapped(req, res, next) {
    try { await handler(req, res); } catch (e) { next(e); }
  };
}

export { compose, asMiddleware };
