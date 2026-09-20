/**
 * model.js
 * Changes when: the model definition (weights, architecture) changes.
 */

import { Vessert } from '../vessert/Vessert.js';

Vessert.setSeed(42);
const INPUT_DIM = 3, OUTPUT_DIM = 2;
const W = Vessert.randn([INPUT_DIM, OUTPUT_DIM]);
const b = Vessert.zeros([OUTPUT_DIM]);

/**
 * Compute softmax(logits) for one input vector.
 * @param {number[]} inputs
 * @returns {Vessert}
 */
function predict(inputs) {
  if (!Array.isArray(inputs) || inputs.length !== INPUT_DIM) {
    throw new Error(`predict: input must be an array of length ${INPUT_DIM}`);
  }
  const x = Vessert.from(inputs, [1, INPUT_DIM]);
  const logits = x.matmul(W).add(b);
  const shifted = logits.sub(logits.max());
  const exps = shifted.exp();
  return exps.div(exps.sum());
}

/**
 * @returns {{inputDim:number, outputDim:number, weightShape:number[], biasShape:number[]}}
 */
function info() {
  return { inputDim: INPUT_DIM, outputDim: OUTPUT_DIM, weightShape: W.shape, biasShape: b.shape };
}

export { predict, info };
