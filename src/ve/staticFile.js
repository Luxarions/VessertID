/**
 * staticFile.js
 * Changes when: how static files are served changes.
 * @memberof ve
 */

import fs from 'node:fs';
import path from 'node:path';
import { resolveSafePath, verifyRealPath } from './path.js';
import { sendText, sendError, sendNotFound, sendForbidden } from './respond.js';

/**
 * @memberof ve
 * @type {Readonly<Record<string,string>>}
 */
const MIME_TYPES = Object.freeze({
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.mjs':  'application/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.ico':  'image/x-icon',
  '.txt':  'text/plain; charset=utf-8',
  '.map':  'application/json; charset=utf-8',
});

/**
 * @memberof ve
 */
async function serveStaticFile(req, res, options) {
  const { rootDir, indexFallback, onForbidden } = options;

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    sendText(res, 405, '405 Method Not Allowed');
    return;
  }

  const lexical = resolveSafePath(rootDir, req.url, indexFallback);
  if (!lexical) {
    if (onForbidden) onForbidden(req.url);
    sendForbidden(res);
    return;
  }

  const real = await verifyRealPath(rootDir, lexical);
  if (!real) { sendNotFound(res); return; }

  let stat;
  try { stat = await fs.promises.stat(real); } catch { sendNotFound(res); return; }
  if (!stat.isFile()) { sendNotFound(res); return; }

  const ext = path.extname(real).toLowerCase();
  const type = MIME_TYPES[ext] || 'application/octet-stream';

  res.writeHead(200, {
    'Content-Type': type,
    'Content-Length': stat.size,
    'Cache-Control': 'no-cache',
  });

  if (req.method === 'HEAD') { res.end(); return; }

  const stream = fs.createReadStream(real);
  stream.on('error', (e) => { if (!res.headersSent) sendError(res, e); else res.end(); });
  stream.pipe(res);
}

export { serveStaticFile, MIME_TYPES };
