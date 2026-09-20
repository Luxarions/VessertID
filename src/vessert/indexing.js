/**
 * indexing.js
 * Changes when: element access (at/sliceAxis) changes.
 * @memberof Vessert
 */

import { Dtype } from './dtype.js';
import { stridesOf, offsetOf, makeHandle } from './handle.js';

/** @memberof Vessert */
function at(t, idx) {
  const off = offsetOf(idx, stridesOf(t.shape));
  return { data: new (Dtype.ctor(t.dtype))([t.data[off]]), shape: [], dtype: t.dtype };
}

/** @memberof Vessert */
function sliceAxis(t, axis, start, stop, step = 1) {
  const ndim = t.shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`sliceAxis: axis ${axis} out of range`);
  start = start ?? 0; stop = stop ?? t.shape[axis];
  if (start < 0) start += t.shape[axis];
  if (stop < 0) stop += t.shape[axis];
  const len = Math.max(0, Math.ceil((stop - start) / step));
  const outShape = t.shape.slice();
  outShape[axis] = len;
  const out = makeHandle(outShape, t.dtype, 0);
  const st = stridesOf(t.shape);
  const idx = new Array(ndim).fill(0);
  const src = new Array(ndim).fill(0);
  for (let i = 0; i < out.data.length; i++) {
    for (let d = 0; d < ndim; d++) src[d] = idx[d];
    src[axis] = start + idx[axis] * step;
    out.data[i] = t.data[offsetOf(src, st)];
    for (let d = ndim - 1; d >= 0; d--) { if (++idx[d] < outShape[d]) break; idx[d] = 0; }
  }
  return out;
}

export { at, sliceAxis };
