
/**
 * Dependencies.
 */

var sample = require('..');
var assert = require('assert');

/**
 * Tests.
 */

describe('sample()', function() {
  it('should be a function', function() {
    assert.equal(typeof sample, 'function');
  });

  it('should return a number', function() {
    assert.equal(typeof sample(), 'number');
  });
});