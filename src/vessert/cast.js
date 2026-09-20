/**
 * cast.js
 * Changes when: dtype conversion rules change.
 * @memberof Vessert
 */

import { makeHandle } from './handle.js';

/** @memberof Vessert */
function astype(t, dtype) {
  const out = makeHandle(t.shape, dtype, 0);
  out.data.set(t.data);
  return out;
}

export { astype };
