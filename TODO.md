# TODO

experiment with ridiculously large hidden sizes and letter sizes
does it make sense for hidden size to be bigger than letter size?

## Inbox

* [IN PROGRESS] replace all for(;;) loops with for...of range
* Next: focus on computeCost and all related code
* Need to reorganize model...too big. Maybe split out everything not related to creating/loading/serializing...or split out to/from json into a separate file?
* Maybe another data structure to represent a layer..and/or one to represent a DNN (could be called network, graph, whatever)
* Output epoch so caller can anneal learning rate per epoch if desired

## Restructure

* [IN PROGRESS] Restructuring and improving Model
  * [IN PROGRESS] costFunc -> forwardIndex -> forwardRNN / LSTM (prev?) - how does lh.o.dw = ... have any effect? So much reliance on weird side-effects...uggh
  * Separate side-effecty stuff from pure functions. Convert as much as possible to pure functions.
    * Notate side-effects
      * forwardRNN/LSTM side-effects the passed in Graph, which is then used in Model to do backprop
      * forwardRNN/LSTM side-effects prev.c, prev.h
      * optimize side-effects passed in stepCache...maybe just make immutable
* review each module
  * Model
    * computeCost
      * understand math
      * understand how lh.o.dw is having any effect
  * Solver
  * RNN
  * Utils
  * Mat (done)
  * Graph (done)
    * Consider moving some of this code to other modules
* get a solid understanding of relationships between all modules and how forward/backward prop works in this code, then consider major restructuring

## Data Augmentation

* Train a model, then get tons of samples, then run through grammarly, then add to haiku data

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
