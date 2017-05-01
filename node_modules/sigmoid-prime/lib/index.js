
/**
 * Expose `sigmoidPrime`.
 */

module.exports = sigmoidPrime;

/**
 * Derivative of the sigmoid function.
 *
 * - Used to calculate the deltas in neural networks.
 *
 * @param {Number} z
 */

function sigmoidPrime(z) {
  return Math.exp(-z) / Math.pow(1 + Math.exp(-z), 2);
}
