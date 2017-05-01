
/**
 * Dependencies.
 */

var sigmoidPrime = require('..');
var assert = require('assert');

/**
 * Tests.
 */

describe('sigmoidPrime()', function() {
  it('should be a function', function() {
    assert.equal(typeof sigmoidPrime, 'function');
  });

  it('should calculate the derivative of the sigmoid for negative numbers', function() {
    assert.equal(sigmoidPrime(-0.29), 0.24481655684290712);
  });

  it('should calculate the derivative of the sigmoid for positive numbers', function() {
    assert.equal(sigmoidPrime(0.98), 0.19842185807035076);
  });
});
