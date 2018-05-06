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
Imported only by RNN

# Solver

Does actual updating of weights/params
Only imported externally (check which other things are import externally)
