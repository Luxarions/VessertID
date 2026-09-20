/**
 * elementwise.js
 * Changes when: per-element operations (binary/unary/where) change.
 * @memberof Vessert
 */

import { offsetOf, bcastShape, bcastStrides, makeHandle } from './handle.js';

/** @memberof Vessert */
function binary(a, b, fn) {
  const shape = bcastShape(a.shape, b.shape);
  const sa = bcastStrides(a.shape, shape);
  const sb = bcastStrides(b.shape, shape);
  const out = makeHandle(shape, a.dtype, 0);
  const idx = new Array(shape.length).fill(0);
  for (let i = 0; i < out.data.length; i++) {
    out.data[i] = fn(a.data[offsetOf(idx, sa)], b.data[offsetOf(idx, sb)]);
    for (let d = shape.length - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
  }
  return out;
}

/** @memberof Vessert */
function unary(t, fn) {
  const out = makeHandle(t.shape, t.dtype, 0);
  for (let i = 0; i < t.data.length; i++) out.data[i] = fn(t.data[i]);
  return out;
}

/** @memberof Vessert */
function where(cond, a, b) {
  const shape = bcastShape(bcastShape(cond.shape, a.shape), b.shape);
  const sc = bcastStrides(cond.shape, shape);
  const sa = bcastStrides(a.shape, shape);
  const sb = bcastStrides(b.shape, shape);
  const out = makeHandle(shape, a.dtype, 0);
  const idx = new Array(shape.length).fill(0);
  for (let i = 0; i < out.data.length; i++) {
    const c = cond.data[offsetOf(idx, sc)];
    out.data[i] = c !== 0 ? a.data[offsetOf(idx, sa)] : b.data[offsetOf(idx, sb)];
    for (let d = shape.length - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
  }
  return out;
}

export { binary, unary, where };
