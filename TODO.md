# TODO

## Restructure

* Create a "Model" class and hang everything that gets set in initVocab on it, as well as obvious stuff
  * Maybe model can also have solver and graph etc so that model gives you the full API and you don't have to piece things together
  * review usages of rnn from recurrent after doing this
* review each module
  * Graph
    * Maybe use generator / call stack type data structure here?
    * Review that other guy's lib
  * RNN
    * Create an API so that code in recurrent pages/index can be generalized and moved into here
  * Solver
  * utils
* functions (or generators?) and object factories instead of classes
* optimize...maybe bring in a math lib (math.js? http://mathjs.org/docs/datatypes/matrices.html)

## Tests

* Jest

## Next

* Convert to tensorflow.js

## Random Ideas

* Mat is the basic data structure...is there a better way to do it?

// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something

Use same tags as this https://github.com/mvrahden/recurrent-js
