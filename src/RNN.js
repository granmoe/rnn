import Mat, { RandMat } from './Mat'

// Prob start analysis of this module with RNN since it's simpler
// split RNN and LSTM into separate files?
// create Model class that ties everything together (and generalizes the stuff in recurrent/pages/index)

export function initLSTM(inputSize, hiddenSizes, outputSize) {
  return hiddenSizes.reduce((model, hiddenSize, index, hiddenSizes) => {
    const prevSize = index === 0 ? inputSize : hiddenSizes[index - 1]

    // gates parameters
    model['Wix' + index] = new RandMat(hiddenSize, prevSize, 0.08)
    model['Wih' + index] = new RandMat(hiddenSize, hiddenSize, 0.08)
    model['bi' + index] = new Mat(hiddenSize, 1)
    model['Wfx' + index] = new RandMat(hiddenSize, prevSize, 0.08)
    model['Wfh' + index] = new RandMat(hiddenSize, hiddenSize, 0.08)
    model['bf' + index] = new Mat(hiddenSize, 1)
    model['Wox' + index] = new RandMat(hiddenSize, prevSize, 0.08)
    model['Woh' + index] = new RandMat(hiddenSize, hiddenSize, 0.08)
    model['bo' + index] = new Mat(hiddenSize, 1)

    // cell write params
    model['Wcx' + index] = new RandMat(hiddenSize, prevSize, 0.08)
    model['Wch' + index] = new RandMat(hiddenSize, hiddenSize, 0.08)
    model['bc' + index] = new Mat(hiddenSize, 1)

    // decoder params
    model['Whd'] = new RandMat(outputSize, hiddenSize, 0.08)
    model['bd'] = new Mat(outputSize, 1)

    // letter embedding vectors
    model['Wil'] = new RandMat(outputSize, inputSize, 0, 0.08)

    return model
  }, {})
}

// TODO: further refactoring here and make sure to understand everything
export function forwardLSTM(graph, model, x, prev) {
  // forward prop for a single tick of LSTM
  // model contains LSTM parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden and cell
  // from previous iteration

  let hiddenPrevs, cellPrevs
  if (typeof prev.h === 'undefined') {
    hiddenPrevs = model.hyperParams.hiddenSizes.map(
      hiddenSize => new Mat(hiddenSize, 1),
    )
    cellPrevs = [...hiddenPrevs]
  } else {
    hiddenPrevs = prev.h
    cellPrevs = prev.c
  }

  const { hidden, cell } = model.hyperParams.hiddenSizes.reduce(
    (result, hiddenSize, index, hiddenSizes) => {
      let inputVector = index === 0 ? x : result.hidden[index - 1]
      let hiddenPrev = hiddenPrevs[index]
      let cellPrev = cellPrevs[index]

      // input gate
      let h0 = graph.mul(model['Wix' + index], inputVector)
      let h1 = graph.mul(model['Wih' + index], hiddenPrev)
      let inputGate = graph.sigmoid(
        graph.add(graph.add(h0, h1), model['bi' + index]),
      )

      // forget gate
      let h2 = graph.mul(model['Wfx' + index], inputVector)
      let h3 = graph.mul(model['Wfh' + index], hiddenPrev)
      let forgetGate = graph.sigmoid(
        graph.add(graph.add(h2, h3), model['bf' + index]),
      )

      // output gate
      let h4 = graph.mul(model['Wox' + index], inputVector)
      let h5 = graph.mul(model['Woh' + index], hiddenPrev)
      let outputGate = graph.sigmoid(
        graph.add(graph.add(h4, h5), model['bo' + index]),
      )

      // write operation on cells
      let h6 = graph.mul(model['Wcx' + index], inputVector)
      let h7 = graph.mul(model['Wch' + index], hiddenPrev)
      let cellWrite = graph.tanh(
        graph.add(graph.add(h6, h7), model['bc' + index]),
      )

      // compute new cell activation
      let retainCell = graph.eltmul(forgetGate, cellPrev) // what do we keep from cell
      let writeCell = graph.eltmul(inputGate, cellWrite) // what do we write to cell
      let cellD = graph.add(retainCell, writeCell) // new cell contents

      // compute hidden state as gated, saturated cell activations
      let hiddenD = graph.eltmul(outputGate, graph.tanh(cellD))

      result.hidden.push(hiddenD)
      result.cell.push(cellD)
      // return [[...hidden, hiddenD], [...cell, cellD]]
      return result
    },
    { hidden: [], cell: [] },
  )

  // one decoder to outputs at end
  let output = graph.add(
    graph.mul(model['Whd'], hidden[hidden.length - 1]),
    model['bd'],
  )

  // return cell memory, hidden representation and output
  return { h: hidden, c: cell, o: output }
}

export function initRNN(inputSize, hiddenSizes, outputSize) {
  const model = hiddenSizes.reduce((model, hiddenSize, index, hiddenSizes) => {
    const prevSize = index === 0 ? inputSize : hiddenSizes[index - 1]

    model['Wxh' + index] = new RandMat(hiddenSize, prevSize, 0.08)
    model['Whh' + index] = new RandMat(hiddenSize, hiddenSize, 0.08)
    model['bhh' + index] = new Mat(hiddenSize, 1)
  }, {})

  // decoder params
  model['Whd'] = new RandMat(
    outputSize,
    hiddenSizes[hiddenSizes.length - 1],
    0.08,
  )
  model['bd'] = new Mat(outputSize, 1)

  // letter embedding vectors
  model['Wil'] = new RandMat(outputSize, inputSize, 0, 0.08)

  return model
}

export function forwardRNN(graph, model, x, prev) {
  // forward prop for a single tick of RNN
  // model contains RNN parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden activations from last step

  // TODO: prev.h is the only thing used from prev...just pass in prev.h directly and use default value

  let hiddenPrevs
  if (typeof prev.h === 'undefined') {
    hiddenPrevs = model.hyperParams.hiddenSizes.map(
      hiddenSize => new Mat(hiddenSize, 1),
    )
  } else {
    hiddenPrevs = prev.h
  }

  const h = model.hyperParams.hiddenSizes.reduce(
    (result, hiddenSize, index) => {
      let inputVector = index === 0 ? x : result[index - 1]
      let hiddenPrev = hiddenPrevs[index]

      let h0 = graph.mul(model['Wxh' + index], inputVector)
      let h1 = graph.mul(model['Whh' + index], hiddenPrev)

      return graph.relu(graph.add(graph.add(h0, h1), model['bhh' + index]))
    },
    [],
  )

  // one decoder to outputs at end
  const o = graph.add(graph.mul(model['Whd'], h[h.length - 1]), model['bd'])

  // return cell memory, hidden representation and output
  return { h, o }
}
