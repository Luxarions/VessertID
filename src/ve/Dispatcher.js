/**
 * Dispatcher.js
 * Changes when: the flow of a single request from arrival to handler/static
 * changes.
 * @memberof ve
 *
 * Every failure stage — synchronous hook, async middleware, handler, terminal
 * static — routes through a single `fail()` so nothing leaks as an unhandled
 * rejection or an uncaught exception.
 */

import { parseUrl } from './parseUrl.js';
import { compose } from './middleware.js';
import { toHttpError } from './HttpError.js';
import { sendText, sendError } from './respond.js';
import { serveStaticFile } from './staticFile.js';

/**
 * @memberof ve
 * @class
 */
class Dispatcher {
  constructor({ router, middlewares, rootDir, indexFallback, bodyLimit, hooks }) {
    this._router = router;
    this._middlewares = middlewares;
    this._rootDir = rootDir;
    this._indexFallback = indexFallback;
    this._bodyLimit = bodyLimit;
    this._hooks = hooks || {};
  }

  handle(req, res) {
    const hooks = this._hooks;

    const fail = (err) => {
      const httpErr = toHttpError(err);
      if (hooks.onError) {
        try { hooks.onError(httpErr, req); }
        catch (hookErr) {
          process.stderr.write(`dispatcher: onError hook threw: ${hookErr.message}\n`);
        }
      }
      if (!res.headersSent) sendError(res, httpErr);
      else res.end();
    };

    let match;
    try {
      if (hooks.onRequest) hooks.onRequest(req, res);

      const { pathname } = parseUrl(req);
      match = this._router.match(req.method, pathname);
      req.route = match && match.handler
        ? { params: match.params, pattern: match.pattern }
        : null;
    } catch (err) { fail(err); return; }

    const terminal = async (req, res) => {
      if (match && match.handler) { await match.handler(req, res); return; }
      if (match && match.allowed) {
        res.setHeader('Allow', match.allowed.join(', '));
        sendText(res, 405, '405 Method Not Allowed');
        return;
      }
      await serveStaticFile(req, res, {
        rootDir: this._rootDir,
        indexFallback: this._indexFallback,
        onForbidden: hooks.onForbidden,
      });
    };

    const chain = compose([...this._middlewares, terminal]);
    chain(req, res).catch(fail);
  }
}

export { Dispatcher };
