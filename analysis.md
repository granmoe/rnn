# Mat usages

## Graph

new Mat(rows, cols) x 2

// these are only used a few times...maybe not really needed?
m.clone().updateW() x 2
updateMats x 2

## RNN

new RandMat(rows, cols, 0.08) x 11
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

* create
* load
* serialize

* train

  * forward
  * backward
  * optimize?

* Mat
* Graph is the odd man out

Create graph of imports (use webpack analyzer) and reorganize code accordingly
