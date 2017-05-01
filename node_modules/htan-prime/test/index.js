
/**
 * Dependencies.
 */

var htanPrime = require('..');
var assert = require('assert');

/**
 * Tests.
 */

describe('htanPrime()', function() {
  it('should be a function', function() {
    assert.equal(typeof htanPrime, 'function');
  });

  it('should calculate the derivative of the hyperbolic tangent for negative numbers', function() {
    assert.equal(htanPrime(-0.89), 0.49391895832869004);
  });

  it('should calculate the derivative of the hyperbolic tangent for positive numbers', function() {
    assert.equal(htanPrime(0.72), 0.6194229120237337);
  });
});