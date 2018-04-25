# TODO

## Restructure

* review each module
  * Mat API:
    * Method for updating weights can either take new weights array or a map function?
    * Constructor can optionally take weights?
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
// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something