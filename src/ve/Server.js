/**
 * Server.js
 * Changes when: the HTTP server lifecycle (listen/close/timeout) changes.
 * Receives a request handler as an argument. Knows nothing about routing,
 * static files, or middleware.
 * @memberof ve
 */

import http from 'node:http';
import { EventEmitter } from './EventEmitter.js';
import { installSignals } from './signals.js';

/**
 * @memberof ve
 * @class
 * @extends EventEmitter
 */
class Server extends EventEmitter {
  constructor(handler, options = {}) {
    super();
    if (typeof handler !== 'function') throw new Error('Server: handler must be a function');
    this._handler = handler;
    this._http = http.createServer((req, res) => this._handler(req, res));
    this._http.requestTimeout = options.requestTimeout ?? 30_000;
    this._http.headersTimeout = options.headersTimeout ?? 10_000;
    this._uninstallSignals = null;
  }

  address() { return this._http.address(); }

  listen(port = 3000, host = '0.0.0.0') {
    return new Promise((resolve, reject) => {
      const onErr = (err) => { this._http.off('listening', onOk); reject(err); };
      const onOk = () => {
        this._http.off('error', onErr);
        const addr = this._http.address();
        const boundPort = typeof addr === 'object' && addr ? addr.port : port;
        this.emit('listening', { port: boundPort, host });
        resolve({ port: boundPort, host, server: this._http });
      };
      this._http.once('error', onErr);
      this._http.once('listening', onOk);
      this._http.listen(port, host);
    });
  }

  close() {
    return new Promise((resolve, reject) => {
      if (this._uninstallSignals) { this._uninstallSignals(); this._uninstallSignals = null; }
      this._http.close((err) => (err ? reject(err) : resolve()));
      if (typeof this._http.closeIdleConnections === 'function') this._http.closeIdleConnections();
    });
  }

  installSignals(options = {}) {
    if (this._uninstallSignals) return this;
    this._uninstallSignals = installSignals(async (sig) => {
      this.emit('shutdown', sig);
      await this.close();
    }, options);
    return this;
  }
}

export { Server };
