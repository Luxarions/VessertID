/**
 * handle.js
 * Changes when: shape arithmetic (numel, strides, broadcasting) changes.
 * @memberof Vessert
 */

import { Dtype } from './dtype.js';

/** @memberof Vessert */
function numel(shape) {
  let n = 1;
  for (let i = 0; i < shape.length; i++) n *= shape[i];
  return n;
}

/** @memberof Vessert */
function stridesOf(shape) {
  const n = shape.length;
  const out = new Array(n);
  let acc = 1;
  for (let i = n - 1; i >= 0; i--) { out[i] = acc; acc *= shape[i]; }
  return out;
}

/** @memberof Vessert */
function offsetOf(idx, st) {
  let off = 0;
  for (let i = 0; i < idx.length; i++) off += idx[i] * st[i];
  return off;
}

/** @memberof Vessert */
function bcastShape(a, b) {
  const n = Math.max(a.length, b.length);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const ia = a.length - n + i, ib = b.length - n + i;
    const va = ia >= 0 ? a[ia] : 1, vb = ib >= 0 ? b[ib] : 1;
    if (va !== vb && va !== 1 && vb !== 1) throw new Error(`shape mismatch: [${a}] vs [${b}]`);
    out[i] = va > vb ? va : vb;
  }
  return out;
}

/** @memberof Vessert */
function bcastStrides(shape, target) {
  const off = target.length - shape.length;
  const s = stridesOf(shape);
  const out = new Array(target.length);
  for (let i = 0; i < target.length; i++) {
    out[i] = i < off ? 0 : (shape[i - off] === 1 ? 0 : s[i - off]);
  }
  return out;
}

/** @memberof Vessert */
function makeHandle(shape, dtype, fill) {
  const data = new (Dtype.ctor(dtype))(numel(shape));
  if (fill !== undefined) data.fill(fill);
  return { data, shape: shape.slice(), dtype };
}

export { numel, stridesOf, offsetOf, bcastShape, bcastStrides, makeHandle };
