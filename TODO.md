# TODO

## Restructure

* [IN PROGRESS] Figure out a way to get deterministic output from model to facilitate testing, expose predictSentence, costFunc, etc in API (prob would be much easier after cleaning these up)
* [IN PROGRESS] Restructuring and improving Model
  * put forward method on model (diff for RNN and LSTM) and update call sites
  * Separate side-effecty stuff from pure functions. Convert as much as possible to pure functions.
    * Notate side-effects
* review each module
  * Graph
    * Consider moving some of this code to other modules
  * Solver
  * utils
* Consumer can pass in perplexity when sampling

## Bugs

* If charCountThreshold is > 1, need to filter out excluded chars from input sentences, otherwise shit blows up (text model doesn't have all the chars that are in input). I.e., filter input to include only chars in text model.

## Ideas

* functions (or generators?) and object factories instead of classes
* optimize...maybe bring in a math lib (math.js? http://mathjs.org/docs/datatypes/matrices.html)
* Maybe use generator / call stack type data structure for Graph

## Optimizations

* use web worker https://survivejs.com/webpack/techniques/web-workers/
* checkpoint callback

## Next

* Convert to tensorflow.js

## Random Ideas

* Mat is the basic data structure...is there a better way to do it?

// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something

Use same tags as this https://github.com/mvrahden/recurrent-js

// TODO: try messing with charCountThreshold sometime
