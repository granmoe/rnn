# TODO

## Restructure

* implement from/to JSON, use this to set up snapshot tests
* [IN PROGRESS] Restructuring and improving Model
  * put forward method on model (diff for RNN and LSTM) and update call sites
  * Separate side-effecty stuff from pure functions. Convert as much as possible to pure functions.
* review each module
  * Graph
    * Consider moving some of this code to other modules
  * Solver
  * utils
* Consumer can pass in perplexity when sampling

## Ideas

* functions (or generators?) and object factories instead of classes
* optimize...maybe bring in a math lib (math.js? http://mathjs.org/docs/datatypes/matrices.html)
* Maybe use generator / call stack type data structure for Graph
* Any other way to implement loop other than setTimeout?

## Next

* Convert to tensorflow.js

## Random Ideas

* Mat is the basic data structure...is there a better way to do it?

// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something

Use same tags as this https://github.com/mvrahden/recurrent-js

// TODO: try messing with charCountThreshold sometime
