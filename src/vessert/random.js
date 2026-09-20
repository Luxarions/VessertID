/**
 * random.js
 * Changes when: how random sampling works changes.
 * @memberof Vessert
 */

import { Dtype } from './dtype.js';
import { makeHandle } from './handle.js';
import { nextRand } from './rng.js';

/** @memberof Vessert */
function rand(shape, dtype = Dtype.DEFAULT) {
  const out = makeHandle(shape, dtype, 0);
  for (let i = 0; i < out.data.length; i++) out.data[i] = nextRand();
  return out;
}

/** @memberof Vessert */
function randn(shape, dtype = Dtype.DEFAULT) {
  const out = makeHandle(shape, dtype, 0);
  for (let i = 0; i < out.data.length; i += 2) {
    const u1 = nextRand() || 1e-12, u2 = nextRand();
    const r = Math.sqrt(-2 * Math.log(u1)), t = 2 * Math.PI * u2;
    out.data[i] = r * Math.cos(t);
    if (i + 1 < out.data.length) out.data[i + 1] = r * Math.sin(t);
  }
  return out;
}

export { rand, randn };
