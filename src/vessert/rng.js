/**
 * rng.js
 * Changes when: the RNG algorithm or seed management changes.
 * @memberof Vessert
 */

let _seed = 12345 | 0;

/** @memberof Vessert */
function nextRand() {
  _seed ^= _seed << 13;
  _seed ^= _seed >>> 17;
  _seed ^= _seed << 5;
  return (_seed >>> 0) / 4294967296;
}

/** @memberof Vessert */
function setSeed(s) { _seed = s | 0; }

export { nextRand, setSeed };
