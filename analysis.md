# Mat usages

## Graph

new Mat(rows, cols) x 2

// these are only used a few times...maybe not really needed?
m.clone().updateW() x 2
updateMats x 2

## RNN

randMat(rows, cols, 0.08) x 11
new Mat(foo, 1) x 8

## Solver

new Mat(rows, cols)

## utils

new Mat(rows, cols) // for softmax

# Graph

Does forward/backward prop matrix ops
Tracks what forward ops have been done, exposes a function to run the corresponding backprop based on this

# optimizer

Does actual updating of weights/params

init lstm/rnn

Graph

Mat

forward.js:

computeCost
predictSentence
forwardIndex

Should forwardLSTM and forwardRNN go in forward.js, too? Prob since that's the only place they're imported

=============

* io

  * create
  * load
  * serialize

* train

  * forward
    * computeCost
    * predictSentence
    * forwardIndex (forwardLSTM/RNN)
  * backward (what to do with Graph?)
  * optimize?

* Mat is good on its own (but could rewrite if convert to immut)

* Graph is the odd man out

Create graph of imports (use webpack analyzer) and reorganize code accordingly

RNN.js

// Prob start analysis of this module with RNN since it's simpler
// split RNN and LSTM into separate files?
