import Mat, { RandMat } from './Mat'

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

    return model
  }, {})
}

// TODO: further refactoring here and make sure to understand everything
export function forwardLSTM(G, model, hiddenSizes, x, prev) {
  // forward prop for a single tick of LSTM
  // G is graph to append ops to
  // model contains LSTM parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden and cell
  // from previous iteration

  let hiddenPrevs, cellPrevs
  if (typeof prev.h === 'undefined') {
    hiddenPrevs = hiddenSizes.map(hiddenSize => new Mat(hiddenSize, 1))
    cellPrevs = [...hiddenPrevs]
  } else {
    hiddenPrevs = prev.h
    cellPrevs = prev.c
  }

  const { hidden, cell } = hiddenSizes.reduce(
    (result, hiddenSize, index, hiddenSizes) => {
      let inputVector = index === 0 ? x : result.hidden[index - 1]
      let hiddenPrev = hiddenPrevs[index]
      let cellPrev = cellPrevs[index]

      // input gate
      let h0 = G.mul(model['Wix' + index], inputVector)
      let h1 = G.mul(model['Wih' + index], hiddenPrev)
      let inputGate = G.sigmoid(G.add(G.add(h0, h1), model['bi' + index]))

      // forget gate
      let h2 = G.mul(model['Wfx' + index], inputVector)
      let h3 = G.mul(model['Wfh' + index], hiddenPrev)
      let forgetGate = G.sigmoid(G.add(G.add(h2, h3), model['bf' + index]))

      // output gate
      let h4 = G.mul(model['Wox' + index], inputVector)
      let h5 = G.mul(model['Woh' + index], hiddenPrev)
      let outputGate = G.sigmoid(G.add(G.add(h4, h5), model['bo' + index]))

      // write operation on cells
      let h6 = G.mul(model['Wcx' + index], inputVector)
      let h7 = G.mul(model['Wch' + index], hiddenPrev)
      let cellWrite = G.tanh(G.add(G.add(h6, h7), model['bc' + index]))

      // compute new cell activation
      let retainCell = G.eltmul(forgetGate, cellPrev) // what do we keep from cell
      let writeCell = G.eltmul(inputGate, cellWrite) // what do we write to cell
      let cellD = G.add(retainCell, writeCell) // new cell contents

      // compute hidden state as gated, saturated cell activations
      let hiddenD = G.eltmul(outputGate, G.tanh(cellD))

      result.hidden.push(hiddenD)
      result.cell.push(cellD)
      // return [[...hidden, hiddenD], [...cell, cellD]]
      return result
    },
    { hidden: [], cell: [] },
  )

  // one decoder to outputs at end
  let output = G.add(
    G.mul(model['Whd'], hidden[hidden.length - 1]),
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

  return model
}

export function forwardRNN(G, model, hiddenSizes, x, prev) {
  // forward prop for a single tick of RNN
  // G is graph to append ops to
  // model contains RNN parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden activations from last step

  let hiddenPrevs
  if (typeof prev.h === 'undefined') {
    hiddenPrevs = hiddenSizes.map(hiddenSize => new Mat(hiddenSize, 1))
  } else {
    hiddenPrevs = prev.h
  }

  const h = hiddenSizes.reduce((result, hiddenSize, index) => {
    let inputVector = index === 0 ? x : result[index - 1]
    let hiddenPrev = hiddenPrevs[index]

    let h0 = G.mul(model['Wxh' + index], inputVector)
    let h1 = G.mul(model['Whh' + index], hiddenPrev)

    return G.relu(G.add(G.add(h0, h1), model['bhh' + index]))
  }, [])

  // one decoder to outputs at end
  const o = G.add(G.mul(model['Whd'], h[h.length - 1]), model['bd'])

  // return cell memory, hidden representation and output
  return { h, o }
}
