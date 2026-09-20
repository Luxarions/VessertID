/**
 * gather.js
 * Changes when: how index/gather/scatter/take works changes.
 * @memberof Vessert
 */

import { Dtype } from './dtype.js';
import { numel, makeHandle } from './handle.js';

/** @memberof Vessert */
function indicesOf(x) {
  if (x && x.data) {
    if (x.shape.length !== 1) throw new Error('indicesOf: indices must be 1D');
    return x.data;
  }
  if (Array.isArray(x) || x instanceof Int32Array || x instanceof Uint8Array) return x;
  throw new Error('indicesOf: expected Vessert, Array, or TypedArray');
}

/** @memberof Vessert */
function gather(t, indices, axis) {
  const ndim = t.shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`gather: axis ${axis} out of range`);
  const idx = indicesOf(indices);
  const axisLen = t.shape[axis];
  const outShape = t.shape.slice();
  outShape[axis] = idx.length;
  const out = makeHandle(outShape, t.dtype, 0);
  const outer = numel(t.shape.slice(0, axis));
  const inner = numel(t.shape.slice(axis + 1));
  let dst = 0;
  for (let o = 0; o < outer; o++) {
    for (let k = 0; k < idx.length; k++) {
      let i = idx[k];
      if (i < 0) i += axisLen;
      if (i < 0 || i >= axisLen) throw new Error(`gather: index ${idx[k]} OOB`);
      const src = (o * axisLen + i) * inner;
      for (let m = 0; m < inner; m++) out.data[dst++] = t.data[src + m];
    }
  }
  return out;
}

/** @memberof Vessert */
function scatter(t, indices, values, axis) {
  const ndim = t.shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`scatter: axis ${axis} out of range`);
  const idx = indicesOf(indices);
  const axisLen = t.shape[axis];
  const vShape = values.shape;
  if (vShape.length !== ndim) throw new Error('scatter: values ndim mismatch');
  if (vShape[axis] !== idx.length) throw new Error('scatter: values axis != indices length');
  const out = makeHandle(t.shape, t.dtype, 0);
  out.data.set(t.data);
  const outer = numel(t.shape.slice(0, axis));
  const inner = numel(t.shape.slice(axis + 1));
  let src = 0;
  for (let o = 0; o < outer; o++) {
    for (let k = 0; k < idx.length; k++) {
      let i = idx[k];
      if (i < 0) i += axisLen;
      if (i < 0 || i >= axisLen) throw new Error(`scatter: index ${idx[k]} OOB`);
      const dst = (o * axisLen + i) * inner;
      for (let m = 0; m < inner; m++) out.data[dst + m] = values.data[src++];
    }
  }
  return out;
}

/** @memberof Vessert */
function take(t, indices) {
  const idx = indicesOf(indices);
  const n = t.data.length;
  const out = { data: new (Dtype.ctor(t.dtype))(idx.length), shape: [idx.length], dtype: t.dtype };
  for (let i = 0; i < idx.length; i++) {
    let j = idx[i];
    if (j < 0) j += n;
    if (j < 0 || j >= n) throw new Error(`take: index ${idx[i]} OOB`);
    out.data[i] = t.data[j];
  }
  return out;
}

export { indicesOf, gather, scatter, take };
