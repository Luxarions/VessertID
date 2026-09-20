/**
 * linalg.js
 * Changes when: linear algebra operations change.
 * @memberof Vessert
 */

import { numel, bcastShape, bcastStrides, makeHandle } from './handle.js';

/** @memberof Vessert */
function matmul(a, b) {
  if (a.shape.length === 0 || b.shape.length === 0) throw new Error('matmul: inputs must be at least 1D');
  const aWasVec = a.shape.length === 1, bWasVec = b.shape.length === 1;
  const aShape = aWasVec ? [1, a.shape[0]] : a.shape;
  const bShape = bWasVec ? [b.shape[0], 1] : b.shape;
  const aN = aShape.length, bN = bShape.length;
  const M = aShape[aN - 2], K = aShape[aN - 1];
  const K2 = bShape[bN - 2], N = bShape[bN - 1];
  if (K !== K2) throw new Error(`matmul: [${a.shape}] x [${b.shape}] K mismatch`);

  const aBatch = aShape.slice(0, aN - 2);
  const bBatch = bShape.slice(0, bN - 2);
  const batchShape = bcastShape(aBatch, bBatch);
  const batchSize = numel(batchShape);

  const finalShape = [...batchShape];
  if (!aWasVec) finalShape.push(M);
  if (!bWasVec) finalShape.push(N);

  const out = makeHandle(finalShape, a.dtype, 0);
  const aBS = bcastStrides(aBatch, batchShape);
  const bBS = bcastStrides(bBatch, batchShape);
  const aMS = M * K, bMS = K * N, oMS = M * N;
  const bIdx = new Array(batchShape.length).fill(0);

  for (let bi = 0; bi < batchSize; bi++) {
    let aOff = 0, bOff = 0;
    for (let d = 0; d < batchShape.length; d++) {
      aOff += bIdx[d] * aBS[d];
      bOff += bIdx[d] * bBS[d];
    }
    aOff *= aMS; bOff *= bMS;
    const oOff = bi * oMS;
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const aik = a.data[aOff + i * K + k];
        if (aik === 0) continue;
        for (let j = 0; j < N; j++) out.data[oOff + i * N + j] += aik * b.data[bOff + k * N + j];
      }
    }
    for (let d = batchShape.length - 1; d >= 0; d--) { if (++bIdx[d] < batchShape[d]) break; bIdx[d] = 0; }
  }
  return out;
}

export { matmul };
