# TODOs

- OPT
- IO
- RNN

# In Progress

- Get optimize working
  - graph always predicts char index 0 if backprop and optimization are on
    - gradients must be getting set wrong
      - !!! Grads getting set wrong because in master lh.o is new every time forward is run
  - original version created backprop funcs for each forward run
    - backprop for each run uses different temp mats
    - backprop needs to be different for each forward run so that it backprops the output char
    - maybe just backprop between each character
  - !!! Run forward/backward/optimize completely on each character
    - Could predictSentence be side-effecting output weights?
    - Grads explode as you go back
      - Check if this happens on master
- Then merge optimize -> graph-rewrite -> master
- Are long-range dependencies still working with this refactor? Compare to performance of master
  - ppl seems to be really bad

* Mat/optimize refactor

  - change layer to obj factory?
  - create a cachedGradients field on Mat that tracks what stepCache currently does

# Next

- Maybe just have one train method on graph that runs computeCost, backward, and optimize
- And one predict method

* Bidirectional
* Learning rate annealing
* Renaming

# Inbox

- !!! Don't serialize input. Developer may want to pass in different input during different training sessions anyway.
- Graph / Mat ... immut data that holds pure funcs (that return new instance of self) as opposed to classes that hold/mutate data via self-mutative funcs
- rework Graph
  - Convert graph to object factory, can return runBackprop every time, keeping track of backprop funcs in closure or something
- Maybe another data structure to represent a layer..and/or one to represent a DNN (could be called network, graph, whatever)
- Output epoch so caller can anneal learning rate per epoch if desired
  - Verify decayRate functionality

# Restructure

- [IN PROGRESS] Restructuring and improving Model
  - [IN PROGRESS] costFunc -> forwardIndex -> forwardRNN / LSTM (prev?) - how does lh.o.gradients = ... have any effect? So much reliance on weird side-effects...uggh
  - Separate side-effecty stuff from pure functions. Convert as much as possible to pure functions.
    - Notate side-effects
      - forwardRNN/LSTM side-effects the passed in Graph, which is then used in Model to do backprop
      - forwardRNN/LSTM side-effects prev.c, prev.h
      - optimize side-effects passed in stepCache...maybe just make immutable

# Data Augmentation

- Train a model, then get tons of samples, then run through grammarly (or equivalent), then add to haiku data

# Bugs

- If charCountThreshold is > 1, need to filter out excluded chars from input sentences, otherwise shit blows up (text model doesn't have all the chars that are in input). I.e., filter input to include only chars in text model.

# Ideas

- Bidirectional RNN: https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/191dd7df9cb91ac22f56ed0dfa4a5651e8767a51/2-Figure2-1.png
- functions (or generators?) and object factories instead of classes
- Maybe bring in a math lib (math.js? http://mathjs.org/docs/datatypes/matrices.html)
- Maybe use generator / call stack type data structure for Graph
- Create word-level version of this (basically just tokenize into words using some function and then use words in place of letters...sentence becomes array of words instead of letters)

## Optimizations

- Optimize for perf in node.js env
- Port to other languages. Go?
- use web worker https://survivejs.com/webpack/techniques/web-workers/

## Someday

- Convert to tensorflow.js

## Random Ideas

- Mat is the basic data structure...is there a better way to do it?

// maybe "rowPluck" could be a method on Mat, then can just do
// out = new Mat({ weights: m.rowPluck(index) }) or something

Use same tags as this https://github.com/mvrahden/recurrent-js

try messing with charCountThreshold sometime (need to fix bug above first)
experiment with ridiculously large hidden sizes and letter sizes
does it make sense for hidden size to be bigger than letter size?

layer
forward -> activations
gradients
