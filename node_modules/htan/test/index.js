
/**
 * Dependencies.
 */

var htan = require('..');
var assert = require('assert');

/**
 * Tests.
 */

describe('htan()', function() {
  it('should be a function', function() {
    assert.equal(typeof htan, 'function');
  });

  it('should calculate the hyperbolic tangent for negative numbers', function() {
    assert.equal(htan(-0.8), -0.664036770267849);
  });

  it('should calculate the hyperbolic tangent for positive numbers', function() {
    assert.equal(htan(1.92), 0.9579173059590634);
  });
});