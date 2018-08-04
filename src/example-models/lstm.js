// inputSize = letter vector size, outputSize = num unique chars, charIndex is our input
export default ({ inputSize, hiddenSizes, outputSize, graph }) => charIndex => {
  const x = graph.rowPluck({ rows: outputSize, cols: inputSize }, charIndex) // Wil

  const finalHidden = hiddenSizes.reduce((prevHidden, hiddenSize) => {
    const input = prevHidden || x // output of last layer (but first layer takes input)
    const hiddenPrev = graph.getMat({ rows: hiddenSize, cols: 1, type: 'zeros' })
    const cellPrev = graph.getMat({ rows: hiddenSize, cols: 1, type: 'zeros' })

    // input gate
    const h0 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wix'
      input,
    )
    const h1 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Wih'
      hiddenPrev,
    )
    const inputGate = graph.sigmoid(
      graph.add(graph.add(h0, h1), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bi'
    )

    // forget gate
    const h2 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wfx'
      input,
    )
    const h3 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Wfh'
      hiddenPrev,
    )
    const forgetGate = graph.sigmoid(
      graph.add(graph.add(h2, h3), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bf'
    )

    // output gate
    const h4 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wox'
      input,
    )
    const h5 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Woh'
      hiddenPrev,
    )
    const outputGate = graph.sigmoid(
      graph.add(graph.add(h4, h5), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bo'
    )

    // write operation on cells
    const h6 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wcx'
      input,
    )
    const h7 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Wch'
      hiddenPrev,
    )
    const cellWrite = graph.tanh(
      graph.add(graph.add(h6, h7), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bc'
    )

    // compute new cell activation
    const retainCell = graph.eltmul(forgetGate, cellPrev) // what do we keep from cell
    const writeCell = graph.eltmul(inputGate, cellWrite) // what do we write to cell
    const cellD = graph.add(retainCell, writeCell) // new cell contents

    // compute hidden state as gated, saturated cell activations and pass it to next iteration
    return graph.eltmul(outputGate, graph.tanh(cellD))
  }, null)

  // output, one decoder to outputs at end
  return graph.add(
    graph.mul(
      { rows: outputSize, cols: finalHidden.rows }, // 'Whd'
      finalHidden,
    ),
    { rows: outputSize, cols: 1, type: 'zeros' }, // 'bd
  )
}
