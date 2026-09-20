/**
 * shape.js
 * Changes when: shape-changing operations (reshape/transpose) change.
 * @memberof Vessert
 */

import { numel, makeHandle } from './handle.js';

/** @memberof Vessert */
function reshape(t, shape) {
  if (numel(shape) !== t.data.length) throw new Error(`reshape: [${t.shape}] -> [${shape}] size mismatch`);
  return { data: t.data, shape: shape.slice(), dtype: t.dtype };
}

/** @memberof Vessert */
function transpose2D(t) {
  if (t.shape.length !== 2) throw new Error('transpose2D: input must be 2D');
  const [M, N] = t.shape;
  const out = makeHandle([N, M], t.dtype, 0);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) out.data[j * M + i] = t.data[i * N + j];
  return out;
}

export { reshape, transpose2D };
