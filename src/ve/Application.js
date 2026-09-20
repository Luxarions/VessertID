/**
 * Application.js
 * Changes when: how a request is composed (routes, middleware, hooks) changes.
 * Knows nothing about sockets, listen, or close.
 * @memberof ve
 */

import { EventEmitter } from './EventEmitter.js';
import { Router } from './router.js';
import { Dispatcher } from './Dispatcher.js';
import { DEFAULT_BODY_LIMIT } from './parseBody.js';

/**
 * @memberof ve
 * @class
 * @extends EventEmitter
 */
class Application extends EventEmitter {
  constructor(options = {}) {
    super();
    this._router = new Router();
    this._middlewares = [];
    this._dispatcher = new Dispatcher({
      router: this._router,
      middlewares: this._middlewares,
      rootDir: options.rootDir ?? (typeof process !== 'undefined' && typeof process.cwd === 'function' ? process.cwd() : '.'),
      indexFallback: options.indexFallback ?? 'index.html',
      bodyLimit: options.bodyLimit ?? DEFAULT_BODY_LIMIT,
      hooks: {
        onRequest:   (req, res) => this.emit('request', req, res),
        onError:     (err, req) => {
          if (this.listenerCount('error') > 0) this.emit('error', err, req);
          else if (typeof process !== 'undefined' && process.stderr && typeof process.stderr.write === 'function') {
            process.stderr.write(`app error: ${err.message}\n`);
          } else {
            console.error(`app error: ${err.message}`);
          }
        },
        onForbidden: (url)      => this.emit('forbidden', url),
      },
    });
  }

  use(fn)      { this._middlewares.push(fn); return this; }
  get(p, h)    { this._router.get(p, h);    return this; }
  post(p, h)   { this._router.post(p, h);   return this; }
  put(p, h)    { this._router.put(p, h);    return this; }
  delete(p, h) { this._router.delete(p, h); return this; }

  handle(req, res) { this._dispatcher.handle(req, res); }
}

export { Application };
