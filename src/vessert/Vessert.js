/**
 * Vessert.js
 * Changes when: the public tensor API (methods, properties, static namespaces)
 * changes.
 * @memberof Vessert
 *
 * IMPORTANT — namespace vs. static method invariant
 * ---------------------------------------------------
 * Every `static` method declared on this class MUST return an instance of
 * Vessert. Handle-level operations (operating on plain `{data, shape, dtype}`
 * objects) live under `Vessert.ops`, which is a single property, NOT a set of
 * individual static method names.
 *
 * Do NOT write `Vessert.zeros = F.zeros`. That would silently overwrite the
 * `static zeros` method with a raw factory, and every subsequent call from
 * application code (`Vessert.zeros([2, 3]).matmul(...)`) would throw
 * "matmul is not a function of Object". This was the original bug; the fix is
 * to funnel all handle-level utilities into one namespaced object.
 */

import { Dtype } from './dtype.js';
import { Device } from './device.js';
import { config } from './config.js';
import { setSeed } from './rng.js';

import * as H   from './handle.js';
import * as F   from './factory.js';
import * as E   from './elementwise.js';
import * as L   from './linalg.js';
import * as R   from './reductions.js';
import * as S   from './shape.js';
import * as C   from './cast.js';
import * as I   from './indexing.js';
import * as M   from './manipulation.js';
import * as G   from './gather.js';
import * as RND from './random.js';
import * as CMP from './compare.js';

/**
 * @memberof Vessert
 * @class
 */
class Vessert {
  constructor(handle) { this._h = handle; }

  /* ---- numeric constants ---- */
  static get PI()      { return Math.PI; }
  static get E()       { return Math.E; }
  static get INF()     { return Infinity; }
  static get NEG_INF() { return -Infinity; }
  static get NAN()     { return NaN; }
  static get EPSILON() { return 1.1920929e-7; }

  /* ---- global configuration (mutates the mutable config object) ---- */
  static setBackend(b) { config.backend = b; }
  static getBackend()  { return config.backend; }
  static setDevice(d)  { config.device = d; }
  static getDevice()   { return config.device; }
  static setDefaultDtype(d) {
    if (!Dtype.is(d)) throw new Error(`setDefaultDtype: unknown dtype ${d}`);
    config.defaultDtype = d;
  }
  static getDefaultDtype() { return config.defaultDtype; }
  static setGradEnabled(f) { config.gradEnabled = !!f; }
  static isGradEnabled()   { return config.gradEnabled; }

  /* ---- factories: every one returns a Vessert instance ---- */
  static from(data, shape, dtype = config.defaultDtype) {
    return new Vessert(F.from(data, shape ?? [data.length], dtype));
  }
  static zeros(shape, dtype = config.defaultDtype) { return new Vessert(F.zeros(shape, dtype)); }
  static ones (shape, dtype = config.defaultDtype) { return new Vessert(F.ones (shape, dtype)); }
  static full (shape, v, dtype = config.defaultDtype) { return new Vessert(F.full(shape, v, dtype)); }
  static eye  (n, dtype = config.defaultDtype) { return new Vessert(F.eye(n, dtype)); }
  static arange(start, stop, step, dtype = config.defaultDtype) {
    return new Vessert(F.arange(start, stop, step, dtype));
  }
  static rand (shape, dtype = config.defaultDtype) { return new Vessert(RND.rand (shape, dtype)); }
  static randn(shape, dtype = config.defaultDtype) { return new Vessert(RND.randn(shape, dtype)); }
  static setSeed(s) { setSeed(s); }

  /* ---- combinators: Vessert in, Vessert out ---- */
  static concat(list, axis = 0) {
    if (!list.length) throw new Error('concat: empty list');
    return new Vessert(M.concat(list.map((t) => t._h), axis));
  }
  static cat(list, axis = 0) { return Vessert.concat(list, axis); }
  static stack(list, axis = 0) {
    if (!list.length) throw new Error('stack: empty list');
    return new Vessert(M.stack(list.map((t) => t._h), axis));
  }
  static where(cond, a, b) { return new Vessert(E.where(cond._h, a._h, b._h)); }

  /* ---- properties ---- */
  get shape()    { return this._h.shape; }
  get dtype()    { return this._h.dtype; }
  get size()     { return this._h.data.length; }
  get ndim()     { return this._h.shape.length; }
  get isScalar() { return this._h.data.length === 1; }
  get isVector() { return this._h.shape.length === 1; }
  get isMatrix() { return this._h.shape.length === 2; }
  get T()        { return this.transpose(); }

  /* ---- binary arithmetic ---- */
  add(o)     { return new Vessert(E.binary(this._h, o._h, (a, b) => a + b)); }
  sub(o)     { return new Vessert(E.binary(this._h, o._h, (a, b) => a - b)); }
  mul(o)     { return new Vessert(E.binary(this._h, o._h, (a, b) => a * b)); }
  div(o)     { return new Vessert(E.binary(this._h, o._h, (a, b) => a / b)); }
  pow(o)     { return new Vessert(E.binary(this._h, o._h, Math.pow)); }
  mod(o)     { return new Vessert(E.binary(this._h, o._h, (a, b) => a % b)); }
  maximum(o) { return new Vessert(E.binary(this._h, o._h, Math.max)); }
  minimum(o) { return new Vessert(E.binary(this._h, o._h, Math.min)); }

  /* ---- comparison ---- */
  equal(o)        { return new Vessert(E.binary(this._h, o._h, (a, b) => (a === b ? 1 : 0))); }
  notEqual(o)     { return new Vessert(E.binary(this._h, o._h, (a, b) => (a !== b ? 1 : 0))); }
  greater(o)      { return new Vessert(E.binary(this._h, o._h, (a, b) => (a >  b ? 1 : 0))); }
  greaterEqual(o) { return new Vessert(E.binary(this._h, o._h, (a, b) => (a >= b ? 1 : 0))); }
  less(o)         { return new Vessert(E.binary(this._h, o._h, (a, b) => (a <  b ? 1 : 0))); }
  lessEqual(o)    { return new Vessert(E.binary(this._h, o._h, (a, b) => (a <= b ? 1 : 0))); }

  /* ---- unary ---- */
  neg()         { return new Vessert(E.unary(this._h, (x) => -x)); }
  abs()         { return new Vessert(E.unary(this._h, Math.abs)); }
  sqrt()        { return new Vessert(E.unary(this._h, Math.sqrt)); }
  exp()         { return new Vessert(E.unary(this._h, Math.exp)); }
  log()         { return new Vessert(E.unary(this._h, Math.log)); }
  sin()         { return new Vessert(E.unary(this._h, Math.sin)); }
  cos()         { return new Vessert(E.unary(this._h, Math.cos)); }
  tanh()        { return new Vessert(E.unary(this._h, Math.tanh)); }
  square()      { return new Vessert(E.unary(this._h, (x) => x * x)); }
  relu()        { return new Vessert(E.unary(this._h, (x) => (x > 0 ? x : 0))); }
  sigmoid()     { return new Vessert(E.unary(this._h, (x) => 1 / (1 + Math.exp(-x)))); }
  clip(min, max){ return new Vessert(E.unary(this._h, (x) => Math.min(Math.max(x, min), max))); }
  where(ifTrue, ifFalse) { return new Vessert(E.where(this._h, ifTrue._h, ifFalse._h)); }

  /* ---- reductions ---- */
  sum (axis = null) { return new Vessert(R.reduce(this._h, axis, (a, v) => a + v, 0)); }
  prod(axis = null) { return new Vessert(R.reduce(this._h, axis, (a, v) => a * v, 1)); }
  max (axis = null) { return new Vessert(R.reduce(this._h, axis, (a, v) => (v > a ? v : a), -Infinity)); }
  min (axis = null) { return new Vessert(R.reduce(this._h, axis, (a, v) => (v < a ? v : a),  Infinity)); }
  mean(axis = null) {
    const s = this.sum(axis);
    const n = axis == null
      ? this._h.data.length
      : this._h.shape[axis < 0 ? this._h.shape.length + axis : axis];
    return new Vessert(E.unary(s._h, (x) => x / n));
  }
  argmax(axis = null) { return new Vessert(R.argReduce(this._h, axis, (a, b) => a > b)); }
  argmin(axis = null) { return new Vessert(R.argReduce(this._h, axis, (a, b) => a < b)); }
  cumsum(axis = 0)    { return new Vessert(R.cumsum(this._h, axis)); }

  /* ---- linalg ---- */
  matmul(o) { return new Vessert(L.matmul(this._h, o._h)); }
  dot(o)    { return this.matmul(o); }

  /* ---- shape ---- */
  reshape(shape) { return new Vessert(S.reshape(this._h, shape)); }
  transpose()    { return new Vessert(S.transpose2D(this._h)); }
  flatten()      { return this.reshape([this._h.data.length]); }

  /* ---- join / split (instance-level) ---- */
  concat(list, axis = 0) { return Vessert.concat([this, ...list], axis); }
  stack(list, axis = 0)  { return Vessert.stack([this, ...list], axis); }
  split(sections, axis = 0) { return M.split(this._h, sections, axis).map((h) => new Vessert(h)); }

  /* ---- indexing ---- */
  at(...coords) { return new Vessert(I.at(this._h, coords)); }
  sliceAxis(axis, start, stop, step) { return new Vessert(I.sliceAxis(this._h, axis, start, stop, step)); }
  gather(indices, axis = 0) {
    return new Vessert(G.gather(this._h, indices instanceof Vessert ? indices._h : indices, axis));
  }
  scatter(indices, values, axis = 0) {
    return new Vessert(G.scatter(
      this._h,
      indices instanceof Vessert ? indices._h : indices,
      values instanceof Vessert ? values._h : values,
      axis,
    ));
  }
  take(indices) {
    return new Vessert(G.take(this._h, indices instanceof Vessert ? indices._h : indices));
  }

  /* ---- casting ---- */
  astype(dtype) { return new Vessert(C.astype(this._h, dtype)); }
  to(dtype)     { return this.astype(dtype); }
  float()       { return this.astype('float32'); }
  double()      { return this.astype('float64'); }
  int()         { return this.astype('int32'); }

  /* ---- tensor comparison ---- */
  arrayEqual(o)    { return CMP.arrayEqual(this._h, o._h); }
  allClose(o, tol) { return CMP.allClose(this._h, o._h, tol); }
  isClose(o, tol)  { return this.allClose(o, tol); }

  /* ---- interop ---- */
  toArray() { return Array.from(this._h.data); }
  item() {
    if (this._h.data.length !== 1) throw new Error('item: not a scalar');
    return this._h.data[0];
  }
  clone() { return new Vessert(C.astype(this._h, this._h.dtype)); }
  toString() { return `Vessert(shape=[${this._h.shape}], dtype=${this._h.dtype})`; }
}

/* ---------------------------------------------------------------------------
 * Vessert.ops — handle-level operation namespace.
 *
 * Functions here accept and return plain handles `{ data, shape, dtype }`,
 * NOT Vessert instances. They exist for internal use, extensions, and
 * introspection.
 *
 * Why one property instead of many static methods:
 *   Assigning `Vessert.matmul = L.matmul` would overwrite any future
 *   `static matmul(...)` and would make `Vessert.matmul(a, b)` return a raw
 *   handle instead of a Vessert. Grouping them under `ops` keeps the public
 *   static surface unambiguous and preserves the invariant above.
 * ------------------------------------------------------------------------- */
/**
 * @memberof Vessert
 * @namespace
 */
const ops = Object.freeze({
  ...H, ...F, ...E, ...L, ...R, ...S, ...C, ...I, ...M, ...G, ...RND, ...CMP,
});

Vessert.ops    = ops;
Vessert.dtype  = Dtype;
Vessert.device = Device;
Vessert.config = config;

export { Vessert };
export default Vessert;
