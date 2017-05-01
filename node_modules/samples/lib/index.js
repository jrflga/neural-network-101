
/**
 * Expose `sample`.
 */

module.exports = sample;

/**
 * Generate a random sample from the Guassian distribution.
 *
 * 	- Uses the Box–Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
 */

function sample() {
  return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
}
