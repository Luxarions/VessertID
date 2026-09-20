/**
 * dtype.js
 * Changes when: the dtype registry changes.
 * @memberof Vessert
 */

const CTOR = {
  float32: Float32Array, float64: Float64Array,
  int32: Int32Array, uint8: Uint8Array,
};

/** @memberof Vessert @namespace */
const Dtype = Object.freeze({
  float32: 'float32', float64: 'float64', int32: 'int32', uint8: 'uint8', bool: 'uint8',
  DEFAULT: 'float32',
  is(d) { return typeof d === 'string' && d in CTOR; },
  ctor(d) { const C = CTOR[d]; if (!C) throw new Error(`unknown dtype: ${d}`); return C; },
});

export { Dtype, CTOR };
