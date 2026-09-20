/**
 * ve.js
 * Changes when: the surface of the `ve` namespace changes.
 * @memberof ve
 */

import { EventEmitter } from './EventEmitter.js';
import { HttpError, toHttpError } from './HttpError.js';
import { REASON, reasonOf } from './httpStatus.js';
import { SECURITY_HEADERS, assertHeaderSafe, withSecurityHeaders } from './headers.js';
import { resolveSafePath, verifyRealPath } from './path.js';
import { Router, compilePattern } from './router.js';
import { compose, asMiddleware } from './middleware.js';
import { parseUrl, parseQuery } from './parseUrl.js';
import { parseCookies } from './parseCookies.js';
import { DEFAULT_BODY_LIMIT, readRawBody, readBody } from './parseBody.js';
import { writeHead, sendText, sendJson, sendEmpty, sendError, sendNotFound, sendForbidden } from './respond.js';
import { serveStaticFile, MIME_TYPES } from './staticFile.js';
import { Dispatcher } from './Dispatcher.js';
import { installSignals } from './signals.js';
import { Server } from './Server.js';
import { Application } from './Application.js';

/**
 * @namespace ve
 */
const ve = Object.freeze({
  EventEmitter,
  HttpError, toHttpError,
  REASON, reasonOf,
  SECURITY_HEADERS, assertHeaderSafe, withSecurityHeaders,
  resolveSafePath, verifyRealPath,
  Router, compilePattern,
  compose, asMiddleware,
  parseUrl, parseQuery, parseCookies,
  DEFAULT_BODY_LIMIT, readRawBody, readBody,
  writeHead, sendText, sendJson, sendEmpty, sendError, sendNotFound, sendForbidden,
  serveStaticFile, MIME_TYPES,
  Dispatcher, installSignals, Server, Application,
});

export { ve };
export default ve;
