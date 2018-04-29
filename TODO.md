# TODO

## Restructure

* Create a "Model" class and hang everything that gets set in initVocab on it, as well as obvious stuff
  * Maybe model can also have solver and graph etc so that model gives you the full API and you don't have to piece things together
  * review usages of rnn from recurrent after doing this
* review each module
  * Graph
  * RNN
    * Create an API so that code in recurrent pages/index can be generalized and moved into here
  * Solver
  * utils
* Make recurrent/ an npm module, import here
* Move some of the simple util funcs into a separate utils file
* Maybe move each class into its own file
* pages/index: better data structure...hyper params, input etc need a home
* functions instead of classes
  * generators
* optimize...maybe bring in a math lib (math.js? http://mathjs.org/docs/datatypes/matrices.html)

## Tests

* Jest

## Next

* Convert to tensorflow.js

## Random Ideas

* Mat is the basic data structure...is there a better way to do it?

// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something