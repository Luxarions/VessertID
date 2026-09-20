/**
 * compare.js
 * Changes when: tensor comparison semantics change.
 * @memberof Vessert
 */

/** @memberof Vessert */
function arrayEqual(a, b) {
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) if (a.shape[i] !== b.shape[i]) return false;
  for (let i = 0; i < a.data.length; i++) if (a.data[i] !== b.data[i]) return false;
  return true;
}

/** @memberof Vessert */
function allClose(a, b, tol = 1e-5) {
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) if (a.shape[i] !== b.shape[i]) return false;
  for (let i = 0; i < a.data.length; i++) if (Math.abs(a.data[i] - b.data[i]) > tol) return false;
  return true;
}

export { arrayEqual, allClose };
