put forward method on model (diff for RNN and LSTM) and update call sites

# TODO

## Restructure

* [IN PROGRESS] Begin restructuring and improving Model
  * Need a way to give output to caller (prob callback)
  * Restructure now with no consideration for loading from / dumping to JSON. Should be easy enough to add this later. Need to just get some paint on the canvas.
* [IN PROGRESS] Tests: Run until the word "the" appears, assert that number of iterations is below a certain threshold
  * Maybe same for "to"
  * snapshot testing
* Need deeper understanding of how modules interact in order to further restructure and improve the code
  * Draw all the connections or something
* review each module
  * Graph
    * Consider moving some of this code to other modules
  * RNN
    * Create an API so that code in recurrent pages/index can be generalized and moved into here
  * Solver
  * utils
* After getting a solid understanding of the whole project, create a "Model" class and hang everything that gets set in initVocab on it, as well as obvious stuff
  * Maybe model can also have solver and graph etc so that model gives you the full API and you don't have to piece things together
  * review usages of rnn from recurrent after doing this
* functions (or generators?) and object factories instead of classes
* optimize...maybe bring in a math lib (math.js? http://mathjs.org/docs/datatypes/matrices.html)
* Ideas:
  * Maybe use generator / call stack type data structure for Graph

## Tests

* Jest

## Next

* Convert to tensorflow.js

## Random Ideas

* Mat is the basic data structure...is there a better way to do it?

// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something

Use same tags as this https://github.com/mvrahden/recurrent-js

// TODO: try messing with charCountThreshold sometime
