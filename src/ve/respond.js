/**
 * respond.js
 * Changes when: the response output format (headers, body, error body) changes.
 * @memberof ve
 */

import { toHttpError } from './HttpError.js';
import { reasonOf } from './httpStatus.js';
import { assertHeaderSafe, withSecurityHeaders } from './headers.js';

/**
 * @memberof ve
 */
function writeHead(res, status, headers = {}) {
  if (res.headersSent) return false;
  const merged = withSecurityHeaders(headers);
  for (const [k, v] of Object.entries(merged)) assertHeaderSafe(k, v);
  res.writeHead(status, merged);
  return true;
}

/** @memberof ve */
function sendText(res, status, body, headers = {}) {
  const buf = Buffer.from(String(body));
  writeHead(res, status, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Content-Length': buf.length,
    ...headers,
  });
  res.end(buf);
}

/** @memberof ve */
function sendJson(res, status, data, headers = {}) {
  const buf = Buffer.from(JSON.stringify(data));
  writeHead(res, status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': buf.length,
    ...headers,
  });
  res.end(buf);
}

/** @memberof ve */
function sendEmpty(res, status, headers = {}) {
  writeHead(res, status, headers);
  res.end();
}

/** @memberof ve */
function sendError(res, err) {
  const httpErr = toHttpError(err);
  if (res.headersSent) { res.end(); return; }
  const body = httpErr.expose ? httpErr.message : `${httpErr.status} ${reasonOf(httpErr.status)}`;
  sendText(res, httpErr.status, body, httpErr.headers);
}

/** @memberof ve */ function sendNotFound(res)  { sendText(res, 404, '404 Not Found'); }
/** @memberof ve */ function sendForbidden(res) { sendText(res, 403, '403 Forbidden'); }

export { writeHead, sendText, sendJson, sendEmpty, sendError, sendNotFound, sendForbidden };
