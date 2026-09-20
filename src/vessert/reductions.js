/**
 * reductions.js
 * Changes when: reduction operations (sum/prod/argmax/cumsum) change.
 * @memberof Vessert
 */

import { Dtype } from './dtype.js';
import { numel, stridesOf, offsetOf, makeHandle } from './handle.js';

/** @memberof Vessert */
function reduceAll(t, fn, init) {
  let acc = init;
  for (let i = 0; i < t.data.length; i++) acc = fn(acc, t.data[i]);
  return { data: new (Dtype.ctor(t.dtype))([acc]), shape: [], dtype: t.dtype };
}

/** @memberof Vessert */
function reduceAxis(t, axis, fn, init) {
  const ndim = t.shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`reduce: axis ${axis} out of range`);
  const outShape = t.shape.filter((_, i) => i !== axis);
  const outSize = numel(outShape);
  const outData = new (Dtype.ctor(t.dtype))(outSize).fill(init);
  const inner = stridesOf(outShape);
  const idx = new Array(ndim).fill(0);
  for (let i = 0; i < t.data.length; i++) {
    let oi = 0, k = 0;
    for (let d = 0; d < ndim; d++) { if (d === axis) continue; oi += idx[d] * inner[k++]; }
    outData[oi] = fn(outData[oi], t.data[i]);
    for (let d = ndim - 1; d >= 0; d--) { if (++idx[d] < t.shape[d]) break; idx[d] = 0; }
  }
  return { data: outData, shape: outShape, dtype: t.dtype };
}

/** @memberof Vessert */
function reduce(t, axis, fn, init) {
  return axis == null ? reduceAll(t, fn, init) : reduceAxis(t, axis, fn, init);
}

/** @memberof Vessert */
function argReduce(t, axis, better) {
  if (axis == null) {
    let bi = 0, bv = t.data[0];
    for (let i = 1; i < t.data.length; i++) if (better(t.data[i], bv)) { bv = t.data[i]; bi = i; }
    return { data: new Int32Array([bi]), shape: [], dtype: 'int32' };
  }
  const ndim = t.shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`argReduce: axis ${axis} out of range`);
  const outShape = t.shape.filter((_, i) => i !== axis);
  const outSize = numel(outShape);
  const outData = new Int32Array(outSize);
  const best = new (Dtype.ctor(t.dtype))(outSize);
  const inner = stridesOf(outShape);
  const idx = new Array(ndim).fill(0);
  const seen = new Uint8Array(outSize);
  for (let i = 0; i < t.data.length; i++) {
    let oi = 0, k = 0;
    for (let d = 0; d < ndim; d++) { if (d === axis) continue; oi += idx[d] * inner[k++]; }
    if (!seen[oi] || better(t.data[i], best[oi])) {
      best[oi] = t.data[i]; outData[oi] = idx[axis]; seen[oi] = 1;
    }
    for (let d = ndim - 1; d >= 0; d--) { if (++idx[d] < t.shape[d]) break; idx[d] = 0; }
  }
  return { data: outData, shape: outShape, dtype: 'int32' };
}

/** @memberof Vessert */
function cumsum(t, axis = 0) {
  const ndim = t.shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`cumsum: axis ${axis} out of range`);
  const out = makeHandle(t.shape, t.dtype, 0);
  const st = stridesOf(t.shape);
  const other = new Array(ndim).fill(0);
  const outerSize = numel(t.shape.filter((_, i) => i !== axis));
  for (let o = 0; o < outerSize; o++) {
    let acc = 0;
    for (let k = 0; k < t.shape[axis]; k++) {
      other[axis] = k;
      const off = offsetOf(other, st);
      acc += t.data[off];
      out.data[off] = acc;
    }
    for (let d = ndim - 1; d >= 0; d--) {
      if (d === axis) continue;
      if (++other[d] < t.shape[d]) break;
      other[d] = 0;
    }
  }
  return out;
}

export { reduce, argReduce, cumsum };
