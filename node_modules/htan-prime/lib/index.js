
/**
 * Expose `htanPrime`.
 */

module.exports = htanPrime;

/**
 * Derivative of the hyperbolic tangent function.
 *
 * @param {Number} z
 */

function htanPrime(z) {
  return 1 - Math.pow((Math.exp(2 * z) - 1) / (Math.exp(2 * z) + 1), 2);
}
