
/**
 * Expose `htan`.
 */

module.exports = htan;

/**
 * Hyperbolic tangent function.
 *
 * - Useful for inputs between -1 and 1
 */

function htan(z) {
  return (Math.exp(2 * z) - 1) / (Math.exp(2 * z) + 1);
}