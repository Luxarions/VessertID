/**
 * manipulation.js
 * Changes when: how tensors are joined or split changes.
 * @memberof Vessert
 */

import { numel, makeHandle } from './handle.js';
import { sliceAxis } from './indexing.js';

/** @memberof Vessert */
function concat(ts, axis) {
  if (!ts.length) throw new Error('concat: empty list');
  const ndim = ts[0].shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`concat: axis ${axis} out of range`);
  const outShape = ts[0].shape.slice();
  outShape[axis] = 0;
  for (const t of ts) {
    if (t.shape.length !== ndim) throw new Error('concat: ndim mismatch');
    for (let d = 0; d < ndim; d++) {
      if (d === axis) continue;
      if (t.shape[d] !== ts[0].shape[d]) throw new Error(`concat: dim ${d} mismatch`);
    }
    outShape[axis] += t.shape[axis];
  }
  const out = makeHandle(outShape, ts[0].dtype, 0);
  const outer = numel(outShape.slice(0, axis));
  const inner = numel(outShape.slice(axis + 1));
  let dst = 0;
  for (let o = 0; o < outer; o++) {
    for (const t of ts) {
      const n = t.shape[axis] * inner;
      const src = o * n;
      for (let k = 0; k < n; k++) out.data[dst++] = t.data[src + k];
    }
  }
  return out;
}

/** @memberof Vessert */
function stack(ts, axis) {
  if (!ts.length) throw new Error('stack: empty list');
  const ndim = ts[0].shape.length;
  if (axis < 0) axis += ndim + 1;
  if (axis < 0 || axis > ndim) throw new Error(`stack: axis ${axis} out of range`);
  for (const t of ts) {
    if (t.shape.length !== ndim) throw new Error('stack: ndim mismatch');
    for (let d = 0; d < ndim; d++) {
      if (t.shape[d] !== ts[0].shape[d]) throw new Error('stack: shape mismatch');
    }
  }
  const outShape = ts[0].shape.slice();
  outShape.splice(axis, 0, ts.length);
  const out = makeHandle(outShape, ts[0].dtype, 0);
  const outer = numel(ts[0].shape.slice(0, axis));
  const inner = numel(ts[0].shape.slice(axis));
  let dst = 0;
  for (let o = 0; o < outer; o++) {
    for (const t of ts) {
      const src = o * inner;
      for (let k = 0; k < inner; k++) out.data[dst++] = t.data[src + k];
    }
  }
  return out;
}

/** @memberof Vessert */
function split(t, sections, axis) {
  const ndim = t.shape.length;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) throw new Error(`split: axis ${axis} out of range`);
  let sizes;
  if (Array.isArray(sections)) sizes = sections.slice();
  else {
    if (!Number.isInteger(sections) || sections <= 0) throw new Error('split: sections must be int > 0 or an array');
    const base = Math.floor(t.shape[axis] / sections);
    const rem = t.shape[axis] % sections;
    sizes = Array.from({ length: sections }, (_, i) => base + (i < rem ? 1 : 0));
  }
  let total = 0;
  for (const s of sizes) total += s;
  if (total !== t.shape[axis]) throw new Error(`split: sum ${total} != ${t.shape[axis]}`);
  const outs = [];
  let start = 0;
  for (const s of sizes) { outs.push(sliceAxis(t, axis, start, start + s)); start += s; }
  return outs;
}

export { concat, stack, split };
