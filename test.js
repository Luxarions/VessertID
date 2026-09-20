/**
 * test.js - Pure Unit Test Runner for Library
 * Runs isolated unit tests per file/module and asserts contracts.
 */

import { performance } from 'node:perf_hooks';
import { EventEmitter as NodeEE } from 'node:events';

// Imports: Vessert Units
import { Dtype, CTOR } from './src/vessert/dtype.js';
import { Device } from './src/vessert/device.js';
import { config } from './src/vessert/config.js';
import { numel, stridesOf, offsetOf, bcastShape, bcastStrides, makeHandle } from './src/vessert/handle.js';
import { nextRand, setSeed } from './src/vessert/rng.js';
import { from, zeros, ones, full, eye, arange } from './src/vessert/factory.js';
import { binary, unary, where } from './src/vessert/elementwise.js';
import { matmul } from './src/vessert/linalg.js';
import { reduce, argReduce, cumsum } from './src/vessert/reductions.js';
import { reshape, transpose2D } from './src/vessert/shape.js';
import { astype } from './src/vessert/cast.js';
import { at, sliceAxis } from './src/vessert/indexing.js';
import { concat, stack, split } from './src/vessert/manipulation.js';
import { indicesOf, gather, scatter, take } from './src/vessert/gather.js';
import { rand, randn } from './src/vessert/random.js';
import { arrayEqual, allClose } from './src/vessert/compare.js';
import { Vessert } from './src/vessert/Vessert.js';

// Imports: Ve Units
import { EventEmitter } from './src/ve/EventEmitter.js';
import { HttpError, toHttpError } from './src/ve/HttpError.js';
import { REASON, reasonOf } from './src/ve/httpStatus.js';
import { SECURITY_HEADERS, assertHeaderSafe, withSecurityHeaders } from './src/ve/headers.js';
import { resolveSafePath, verifyRealPath } from './src/ve/path.js';
import { Router, compilePattern } from './src/ve/router.js';
import { compose, asMiddleware } from './src/ve/middleware.js';
import { parseUrl, parseQuery } from './src/ve/parseUrl.js';
import { parseCookies } from './src/ve/parseCookies.js';
import { DEFAULT_BODY_LIMIT, readRawBody, readBody } from './src/ve/parseBody.js';
import { writeHead, sendText, sendJson, sendEmpty, sendError, sendNotFound, sendForbidden } from './src/ve/respond.js';
import { serveStaticFile, MIME_TYPES } from './src/ve/staticFile.js';
import { Dispatcher } from './src/ve/Dispatcher.js';
import { installSignals } from './src/ve/signals.js';
import { Server } from './src/ve/Server.js';
import { Application } from './src/ve/Application.js';
import { ve } from './src/ve/ve.js';

// Imports: App Units & Root
import { predict, info } from './src/app/model.js';
import { mountRoutes } from './src/app/routes.js';
import { renderPreview } from './src/Preview.js';
import * as RootIndex from './src/index.js';

const results = [];
let totalCount = 0;
let passCount = 0;
let failCount = 0;

function runUnit(unitName, testFn) {
  totalCount++;
  const start = performance.now();
  try {
    testFn();
    const duration = (performance.now() - start).toFixed(2);
    passCount++;
    console.log(`\x1b[32m  ✓ [PASS]\x1b[0m ${unitName.padEnd(35)} \x1b[90m(${duration}ms)\x1b[0m`);
    results.push({ name: unitName, status: 'PASS', duration });
  } catch (err) {
    failCount++;
    const duration = (performance.now() - start).toFixed(2);
    console.error(`\x1b[31m  ✗ [FAIL]\x1b[0m ${unitName.padEnd(35)} \x1b[90m(${duration}ms)\x1b[0m`);
    console.error(`\x1b[31m    Error: ${err.message}\x1b[0m`);
    if (err.stack) {
      console.error(`\x1b[90m    ${err.stack.split('\n').slice(1, 4).join('\n    ')}\x1b[0m`);
    }
    results.push({ name: unitName, status: 'FAIL', error: err.message });
  }
}

async function runAsyncUnit(unitName, asyncTestFn) {
  totalCount++;
  const start = performance.now();
  try {
    await asyncTestFn();
    const duration = (performance.now() - start).toFixed(2);
    passCount++;
    console.log(`\x1b[32m  ✓ [PASS]\x1b[0m ${unitName.padEnd(35)} \x1b[90m(${duration}ms)\x1b[0m`);
    results.push({ name: unitName, status: 'PASS', duration });
  } catch (err) {
    failCount++;
    const duration = (performance.now() - start).toFixed(2);
    console.error(`\x1b[31m  ✗ [FAIL]\x1b[0m ${unitName.padEnd(35)} \x1b[90m(${duration}ms)\x1b[0m`);
    console.error(`\x1b[31m    Error: ${err.message}\x1b[0m`);
    results.push({ name: unitName, status: 'FAIL', error: err.message });
  }
}

console.log('\n===============================================================');
console.log('       PENGUJIAN TERMINAL MURNI UNIT-PER-UNIT LIBRARY          ');
console.log('===============================================================\n');

// -------------------------------------------------------------
// [GROUP 1] VESSERT TENSOR ENGINE UNITS
// -------------------------------------------------------------
console.log('\x1b[1m[MODULE GROUP 1: VESSERT TENSOR ENGINE]\x1b[0m');

runUnit('vessert/dtype.js', () => {
  if (!Dtype.is('float32')) throw new Error('Dtype.is float32 failed');
  if (!Dtype.is('int32')) throw new Error('Dtype.is int32 failed');
  if (Dtype.is('unknown_xyz')) throw new Error('Dtype.is unknown returned true');
  if (Dtype.ctor('float32') !== Float32Array) throw new Error('Dtype.ctor mismatch');
});

runUnit('vessert/device.js', () => {
  if (Device.cpu !== 'cpu' || Device.gpu !== 'gpu' || Device.DEFAULT !== 'cpu') {
    throw new Error('Device constant mismatch');
  }
});

runUnit('vessert/config.js', () => {
  if (config.defaultDtype !== 'float32' || config.backend !== 'cpu') {
    throw new Error('config default values mismatch');
  }
});

runUnit('vessert/handle.js', () => {
  if (numel([2, 3, 4]) !== 24) throw new Error('numel failed');
  const st = stridesOf([2, 3]);
  if (st[0] !== 3 || st[1] !== 1) throw new Error('stridesOf failed');
  if (offsetOf([1, 2], st) !== 5) throw new Error('offsetOf failed');
  const bsh = bcastShape([2, 1], [1, 3]);
  if (bsh[0] !== 2 || bsh[1] !== 3) throw new Error('bcastShape failed');
  const h = makeHandle([2, 2], 'float32', 7);
  if (h.data[0] !== 7 || h.data[3] !== 7) throw new Error('makeHandle failed');
});

runUnit('vessert/rng.js', () => {
  setSeed(42);
  const v1 = nextRand();
  setSeed(42);
  const v2 = nextRand();
  if (v1 !== v2) throw new Error('RNG is not deterministic with same seed');
});

runUnit('vessert/factory.js', () => {
  const z = zeros([2, 3]);
  const o = ones([2, 3]);
  const f = full([2, 2], 5);
  const ey = eye(3);
  const ar = arange(0, 5, 1);
  if (z.data.length !== 6 || z.data[0] !== 0) throw new Error('zeros failed');
  if (o.data[0] !== 1) throw new Error('ones failed');
  if (f.data[0] !== 5) throw new Error('full failed');
  if (ey.data[0] !== 1 || ey.data[4] !== 1 || ey.data[1] !== 0) throw new Error('eye failed');
  if (ar.data.length !== 5 || ar.data[4] !== 4) throw new Error('arange failed');
});

runUnit('vessert/elementwise.js', () => {
  const a = from([1, 2, 3], [3]);
  const b = from([10, 20, 30], [3]);
  const sum = binary(a, b, (x, y) => x + y);
  if (sum.data[0] !== 11 || sum.data[2] !== 33) throw new Error('binary add failed');
  const dbl = unary(a, (x) => x * 2);
  if (dbl.data[1] !== 4) throw new Error('unary scale failed');
  const c = from([1, 0, 1], [3], 'uint8');
  const w = where(c, a, b);
  if (w.data[0] !== 1 || w.data[1] !== 20 || w.data[2] !== 3) throw new Error('where failed');
});

runUnit('vessert/linalg.js', () => {
  const a = from([1, 2, 3, 4], [2, 2]);
  const b = from([5, 6, 7, 8], [2, 2]);
  const c = matmul(a, b);
  if (c.data[0] !== 19 || c.data[1] !== 22 || c.data[2] !== 43 || c.data[3] !== 50) {
    throw new Error(`matmul result unexpected: [${c.data}]`);
  }
});

runUnit('vessert/reductions.js', () => {
  const t = from([1, 2, 3, 4], [2, 2]);
  const total = reduce(t, null, (acc, x) => acc + x, 0);
  if (total.data[0] !== 10) throw new Error('reduceAll failed');
  const colSum = reduce(t, 0, (acc, x) => acc + x, 0);
  if (colSum.data[0] !== 4 || colSum.data[1] !== 6) throw new Error('reduceAxis failed');
  const maxIdx = argReduce(from([10, 80, 30], [3]), null, (curr, best) => curr > best);
  if (maxIdx.data[0] !== 1) throw new Error('argReduce failed');
  const cs = cumsum(from([1, 2, 3], [3]), 0);
  if (cs.data[0] !== 1 || cs.data[1] !== 3 || cs.data[2] !== 6) throw new Error('cumsum failed');
});

runUnit('vessert/shape.js', () => {
  const t = from([1, 2, 3, 4, 5, 6], [2, 3]);
  const r = reshape(t, [3, 2]);
  if (r.shape[0] !== 3 || r.shape[1] !== 2) throw new Error('reshape shape mismatch');
  const tr = transpose2D(t);
  if (tr.shape[0] !== 3 || tr.shape[1] !== 2 || tr.data[1] !== 4) throw new Error('transpose2D mismatch');
});

runUnit('vessert/cast.js', () => {
  const f = from([1.7, 3.2], [2], 'float32');
  const i = astype(f, 'int32');
  if (i.dtype !== 'int32' || i.data[0] !== 1 || i.data[1] !== 3) throw new Error('astype float32->int32 failed');
});

runUnit('vessert/indexing.js', () => {
  const m = from([10, 20, 30, 40], [2, 2]);
  const val = at(m, [1, 0]);
  if (val.data[0] !== 30) throw new Error(`at failed: got ${val.data[0]}`);
  const s = sliceAxis(from([10, 20, 30, 40, 50], [5]), 0, 1, 4);
  if (s.data.length !== 3 || s.data[0] !== 20 || s.data[2] !== 40) throw new Error('sliceAxis failed');
});

runUnit('vessert/manipulation.js', () => {
  const a = from([1, 2], [2]);
  const b = from([3, 4], [2]);
  const c = concat([a, b], 0);
  if (c.data.length !== 4 || c.data[2] !== 3) throw new Error('concat failed');
  const st = stack([a, b], 0);
  if (st.shape[0] !== 2 || st.shape[1] !== 2) throw new Error('stack failed');
  const sp = split(c, 2, 0);
  if (sp.length !== 2 || sp[0].data[1] !== 2 || sp[1].data[0] !== 3) throw new Error('split failed');
});

runUnit('vessert/gather.js', () => {
  const t = from([100, 200, 300, 400], [4]);
  const g = gather(t, [3, 0], 0);
  if (g.data[0] !== 400 || g.data[1] !== 100) throw new Error('gather failed');
  const sc = scatter(zeros([4]), [1, 3], from([77, 99], [2]), 0);
  if (sc.data[1] !== 77 || sc.data[3] !== 99) throw new Error('scatter failed');
  const tk = take(t, [2, 1]);
  if (tk.data[0] !== 300 || tk.data[1] !== 200) throw new Error('take failed');
});

runUnit('vessert/random.js', () => {
  const r = rand([5]);
  const rn = randn([5]);
  if (r.data.length !== 5 || rn.data.length !== 5) throw new Error('rand/randn length mismatch');
});

runUnit('vessert/compare.js', () => {
  const a = from([1, 2, 3], [3]);
  const b = from([1, 2, 3], [3]);
  const c = from([1, 2, 4], [3]);
  if (!arrayEqual(a, b)) throw new Error('arrayEqual identity failed');
  if (arrayEqual(a, c)) throw new Error('arrayEqual difference failed');
  if (!allClose(a, b)) throw new Error('allClose identity failed');
});

runUnit('vessert/Vessert.js (Facade)', () => {
  const v1 = Vessert.from([1, 2, 3, 4], [2, 2]);
  const v2 = Vessert.from([5, 6, 7, 8], [2, 2]);
  const v3 = v1.matmul(v2);
  if (v3.shape[0] !== 2 || v3.shape[1] !== 2) throw new Error('Vessert.matmul shape error');
  const added = v1.add(v2);
  if (added.toArray()[0] !== 6) throw new Error('Vessert.add error');
  const sum = added.sum().item();
  if (sum !== (6 + 8 + 10 + 12)) throw new Error(`Vessert.sum.item error: ${sum}`);
  if (v1.size !== 4 || v1.ndim !== 2) throw new Error('Vessert getters error');
});

// -------------------------------------------------------------
// [GROUP 2] VE HTTP ENGINE UNITS
// -------------------------------------------------------------
console.log('\n\x1b[1m[MODULE GROUP 2: VE HTTP ENGINE]\x1b[0m');

runUnit('ve/EventEmitter.js', () => {
  const ee = new EventEmitter();
  let count = 0;
  const listener = (val) => { count += val; };
  ee.on('add', listener);
  ee.emit('add', 5);
  ee.emit('add', 10);
  ee.off('add', listener);
  ee.emit('add', 20);
  if (count !== 15) throw new Error(`EventEmitter listener counter expected 15, got ${count}`);
});

runUnit('ve/HttpError.js', () => {
  const notFound = HttpError.notFound('Resource missing');
  if (notFound.status !== 404 || notFound.message !== 'Resource missing') throw new Error('HttpError.notFound mismatch');
  const badReq = HttpError.badRequest('Invalid payload');
  if (badReq.status !== 400) throw new Error('HttpError.badRequest mismatch');
  const err = toHttpError(new Error('Boom'));
  if (err.status !== 500 || !(err instanceof HttpError)) throw new Error('toHttpError conversion mismatch');
});

runUnit('ve/httpStatus.js', () => {
  if (REASON[200] !== 'OK' || REASON[404] !== 'Not Found' || REASON[500] !== 'Internal Server Error') {
    throw new Error('REASON map mismatch');
  }
  if (reasonOf(201) !== 'Created' || reasonOf(999) !== 'Unknown') {
    throw new Error('reasonOf mismatch');
  }
});

runUnit('ve/headers.js', () => {
  assertHeaderSafe('Content-Type');
  assertHeaderSafe('X-Custom-Header');
  let threw = false;
  try {
    assertHeaderSafe('Bad\r\nInjected');
  } catch {
    threw = true;
  }
  if (!threw) throw new Error('assertHeaderSafe did not detect CRLF injection');

  const secured = withSecurityHeaders({ 'content-type': 'application/json' });
  if (secured['X-Content-Type-Options'] !== 'nosniff') throw new Error('Missing security header');
});

runUnit('ve/path.js', () => {
  const safe = resolveSafePath('/app/project', '/test/file.js');
  if (!safe.startsWith('/app/project')) throw new Error('resolveSafePath escaped root');
  const bad = resolveSafePath('/app/project', '../../../../etc/passwd');
  if (bad !== null) throw new Error('resolveSafePath did not prevent directory traversal');
});

runUnit('ve/router.js', () => {
  const r = new Router();
  r.add('GET', '/users/:userId/profile', () => 'user-profile');
  r.add('POST', '/items', () => 'create-item');

  const match1 = r.match('GET', '/users/usr_42/profile');
  if (!match1 || match1.params.userId !== 'usr_42') throw new Error('Router param matching failed');
  const match2 = r.match('POST', '/items');
  if (!match2 || typeof match2.handler !== 'function') throw new Error('Router static matching failed');
  const match3 = r.match('DELETE', '/items');
  if (match3.handler) throw new Error('Router method mismatch should not return a handler');
  if (!match3.allowed || !match3.allowed.includes('POST')) throw new Error('Router 405 allowed methods missing');
});

await runAsyncUnit('ve/middleware.js', async () => {
  const logs = [];
  const mw1 = async (req, res, next) => { logs.push('m1_enter'); await next(); logs.push('m1_exit'); };
  const mw2 = async (req, res, next) => { logs.push('m2_enter'); await next(); logs.push('m2_exit'); };
  const runner = compose([mw1, mw2]);
  const fakeRes = new NodeEE();
  await runner({}, fakeRes);
  if (logs.join(' -> ') !== 'm1_enter -> m2_enter -> m2_exit -> m1_exit') {
    throw new Error(`compose middleware onion execution mismatch: ${logs.join(' -> ')}`);
  }
});

runUnit('ve/parseUrl.js', () => {
  const parsed = parseUrl({ url: '/api/v1/predict?verbose=1&seed=42' });
  if (parsed.pathname !== '/api/v1/predict') throw new Error(`pathname mismatch: ${parsed.pathname}`);
  if (parsed.query.verbose !== '1' || parsed.query.seed !== '42') throw new Error('query params parsing mismatch');
});

runUnit('ve/parseCookies.js', () => {
  const cookies = parseCookies({ headers: { cookie: 'auth_token=xyz789; theme=dark; session=active' } });
  if (cookies.auth_token !== 'xyz789' || cookies.theme !== 'dark' || cookies.session !== 'active') {
    throw new Error('parseCookies mismatch');
  }
});

runUnit('ve/staticFile.js (MIME types)', () => {
  if (MIME_TYPES['.html'] !== 'text/html; charset=utf-8' || MIME_TYPES['.js'] !== 'application/javascript; charset=utf-8') {
    throw new Error('MIME_TYPES dictionary mismatch');
  }
});

runUnit('ve/Dispatcher.js', () => {
  const d = new Dispatcher({ router: new Router(), middlewares: [] });
  if (typeof d.handle !== 'function') throw new Error('Dispatcher.handle is not a function');
});

runUnit('ve/signals.js', () => {
  if (typeof installSignals !== 'function') throw new Error('installSignals export missing');
});

runUnit('ve/Server.js', () => {
  const s = new Server((req, res) => {});
  if (typeof s.listen !== 'function' || typeof s.close !== 'function') {
    throw new Error('Server lifecycle methods missing');
  }
});

runUnit('ve/Application.js', () => {
  const app = new Application();
  if (typeof app.use !== 'function' || typeof app.get !== 'function' || typeof app.handle !== 'function') {
    throw new Error('Application contract methods missing');
  }
});

runUnit('ve/ve.js (Facade)', () => {
  if (typeof ve !== 'object' || typeof ve.Router !== 'function' || typeof ve.HttpError !== 'function') {
    throw new Error('ve facade namespace missing components');
  }
});

// -------------------------------------------------------------
// [GROUP 3] APPLICATION & ROOT EXPORTS
// -------------------------------------------------------------
console.log('\n\x1b[1m[MODULE GROUP 3: APPLICATION & ROOT EXPORTS]\x1b[0m');

runUnit('app/model.js', () => {
  const mInfo = info();
  if (mInfo.inputDim !== 3 || mInfo.outputDim !== 2) throw new Error('model info mismatch');
  const pred = predict([0.5, -0.2, 0.9]);
  if (pred.shape.length !== 2 || pred.shape[1] !== 2) throw new Error('predict output shape mismatch');
  const probs = pred.toArray();
  const sumProbs = probs[0] + probs[1];
  if (Math.abs(sumProbs - 1.0) > 1e-4) throw new Error(`Softmax probabilities must sum to 1, got ${sumProbs}`);
});

runUnit('app/routes.js', () => {
  const app = new Application();
  mountRoutes(app);
  const m1 = app._router.match('GET', '/health');
  if (!m1 || typeof m1.handler !== 'function') throw new Error('/health route missing');
  const m2 = app._router.match('GET', '/model');
  if (!m2 || typeof m2.handler !== 'function') throw new Error('/model route missing');
  const m3 = app._router.match('POST', '/predict');
  if (!m3 || typeof m3.handler !== 'function') throw new Error('/predict route missing');
});

runUnit('src/index.js (Entrypoint exports)', () => {
  if (typeof RootIndex.Vessert !== 'function') throw new Error('Root Vessert export missing');
  if (typeof RootIndex.Dtype !== 'object') throw new Error('Root Dtype export missing');
  if (typeof RootIndex.Device !== 'object') throw new Error('Root Device export missing');
  if (typeof RootIndex.config !== 'object') throw new Error('Root config export missing');
  if (typeof RootIndex.ve !== 'object') throw new Error('Root ve export missing');
});

runUnit('src/Preview.js (UI Entrypoint)', () => {
  if (typeof renderPreview !== 'function') throw new Error('Root renderPreview export missing');
});

// -------------------------------------------------------------
// SUMMARY REPORT
// -------------------------------------------------------------
console.log('\n===============================================================');
console.log(`  HASIL AKHIR TEST MURNI: ${passCount}/${totalCount} UNIT LULUS`);
if (failCount === 0) {
  console.log('  STATUS: \x1b[32m100% SEMUA UNIT BERFUNGSI SEMPURNA TANPA ERROR\x1b[0m');
} else {
  console.log(`  STATUS: \x1b[31m${failCount} UNIT MENGALAMI KEGAGALAN\x1b[0m`);
}
console.log('===============================================================\n');

if (failCount > 0) process.exit(1);
