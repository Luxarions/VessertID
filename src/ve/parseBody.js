/**
 * parseBody.js
 * Changes when: how request bodies are read and decoded changes.
 * @memberof ve
 *
 * Rejection policy for oversized bodies:
 *   - If Content-Length already exceeds the limit → reject immediately without
 *     reading the rest (client's transfer will be aborted by the socket close
 *     after the 413 response is written).
 *   - If the limit is exceeded while streaming → flag `tooLarge`, drop the
 *     buffered chunks, keep draining until 'end', then reject.
 *
 * This module NEVER calls req.destroy(). The caller (Dispatcher) still owns
 * the socket and must be able to write the 413 response.
 */

import { HttpError } from './HttpError.js';
import { parseQuery } from './parseUrl.js';

/**
 * @memberof ve
 * @type {number}
 */
const DEFAULT_BODY_LIMIT = 1 << 20;

/**
 * @memberof ve
 * @param {*} req
 * @param {number} [limit]
 * @returns {Promise<Buffer>}
 */
function readRawBody(req, limit = DEFAULT_BODY_LIMIT) {
  return new Promise((resolve, reject) => {
    // Fast path: reject from the header before buffering anything.
    const declared = Number(req.headers['content-length']);
    if (Number.isFinite(declared) && declared > limit) {
      req.resume(); // drain so the socket doesn't stall
      reject(HttpError.payloadTooLarge(`body: content-length ${declared} exceeds ${limit}`));
      return;
    }

    const chunks = [];
    let total = 0;
    let done = false;
    let tooLarge = false;

    const finish = (err, buf) => {
      if (done) return;
      done = true;
      req.off('data', onData);
      req.off('end', onEnd);
      req.off('error', onError);
      if (err) reject(err); else resolve(buf);
    };

    const onData = (c) => {
      if (tooLarge) return;      // already over the limit; drop the rest
      total += c.length;
      if (total > limit) {
        tooLarge = true;
        chunks.length = 0;       // release memory
        return;
      }
      chunks.push(c);
    };

    const onEnd = () => {
      if (tooLarge) {
        finish(HttpError.payloadTooLarge(`body: exceeds ${limit} bytes`));
        return;
      }
      finish(null, Buffer.concat(chunks));
    };

    const onError = (e) => finish(HttpError.badRequest(`body: ${e.message}`));

    req.on('data', onData);
    req.on('end', onEnd);
    req.on('error', onError);
  });
}

/**
 * @memberof ve
 * @param {*} req
 * @param {{limit?: number}} [options]
 * @returns {Promise<*|null>}
 */
async function readBody(req, options = {}) {
  const limit = options.limit ?? DEFAULT_BODY_LIMIT;
  const buf = await readRawBody(req, limit);
  if (buf.length === 0) return null;
  const ct = (req.headers['content-type'] || '').split(';')[0].trim().toLowerCase();
  if (ct === 'application/json') {
    try { return JSON.parse(buf.toString('utf8')); }
    catch { throw HttpError.badRequest('body: invalid JSON'); }
  }
  if (ct === 'application/x-www-form-urlencoded') return parseQuery(buf.toString('utf8'));
  if (ct.startsWith('text/')) return buf.toString('utf8');
  return buf;
}

export { DEFAULT_BODY_LIMIT, readRawBody, readBody };
