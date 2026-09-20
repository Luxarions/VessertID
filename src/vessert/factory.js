/**
 * factory.js
 * Changes when: how new tensors are allocated (zeros/ones/full/eye/arange)
 * changes.
 * @memberof Vessert
 */

import { Dtype } from './dtype.js';
import { makeHandle } from './handle.js';

/** @memberof Vessert */
function from(buf, shape, dtype = Dtype.DEFAULT) {
  return { data: new (Dtype.ctor(dtype))(buf), shape: shape.slice(), dtype };
}
/** @memberof Vessert */ function zeros(shape, dtype = Dtype.DEFAULT) { return makeHandle(shape, dtype, 0); }
/** @memberof Vessert */ function ones (shape, dtype = Dtype.DEFAULT) { return makeHandle(shape, dtype, 1); }
/** @memberof Vessert */ function full (shape, v, dtype = Dtype.DEFAULT) { return makeHandle(shape, dtype, v); }

/** @memberof Vessert */
function eye(n, dtype = Dtype.DEFAULT) {
  const t = makeHandle([n, n], dtype, 0);
  for (let i = 0; i < n; i++) t.data[i * n + i] = 1;
  return t;
}

/** @memberof Vessert */
function arange(start, stop, step, dtype = Dtype.float32) {
  if (stop === undefined) { stop = start; start = 0; }
  if (step === undefined || step === null) step = 1;
  const n = Math.max(0, Math.ceil((stop - start) / step));
  const t = makeHandle([n], dtype, 0);
  for (let i = 0; i < n; i++) t.data[i] = start + i * step;
  return t;
}

export { from, zeros, ones, full, eye, arange };
