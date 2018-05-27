# TODO

## Inbox

* Need to reorganize model...too big. Maybe split out everything not related to creating/loading/serializing...or split out to/from json into a separate file?

## Restructure

* [IN PROGRESS] Update tests per change to to/fromJSON
* [IN PROGRESS] Restructuring and improving Model
  * [IN PROGRESS] costFunc -> forwardIndex -> forwardRNN / LSTM (prev?) - whole obj needed for prev? if not, maybe lh.o.... = ... at end of costFunc does nothing and can be deleted? I think only h and c are needed
  * Separate side-effecty stuff from pure functions. Convert as much as possible to pure functions.
    * Notate side-effects
      * forwardRNN/LSTM side-effects the passed in Graph, which is then used in Model to do backprop
* review each module
  * Model
    * costFunc
  * Solver
  * RNN
  * Utils
  * Mat (done)
  * Graph (done)
    * Consider moving some of this code to other modules
* get a solid understanding of relationships between all modules and how forward/backward prop works in this code, then consider major restructuring

## Bugs

* If charCountThreshold is > 1, need to filter out excluded chars from input sentences, otherwise shit blows up (text model doesn't have all the chars that are in input). I.e., filter input to include only chars in text model.

## Ideas

* functions (or generators?) and object factories instead of classes
* Maybe bring in a math lib (math.js? http://mathjs.org/docs/datatypes/matrices.html)
* Maybe use generator / call stack type data structure for Graph
* Create word-level version of this (basically just tokenize into words using some function and then use words in place of letters...sentence becomes array of words instead of letters)

## Optimizations

* checkpoint callback (write to local storage?)
* Optimize for perf in node.js env
* Port to other languages. Go?
* use web worker https://survivejs.com/webpack/techniques/web-workers/

## Next

* Convert to tensorflow.js

## Random Ideas

* Mat is the basic data structure...is there a better way to do it?

// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something

Use same tags as this https://github.com/mvrahden/recurrent-js

// TODO: try messing with charCountThreshold sometime (need to fix bug above first)
