export default ({ graph, outputSize, inputSize, hiddenSizes }) => charIndex => {
  const x = graph.rowPluck({ rows: outputSize, cols: inputSize }, charIndex) // Wil

  const finalHidden = hiddenSizes.reduce((prevHidden, hiddenSize, index) => {
    let inputVector = prevHidden || x
    let hiddenPrev = graph.getMat({ rows: hiddenSize, cols: 1, type: 'zeros' })

    let h0 = graph.mul({ rows: hiddenSize, cols: inputVector.rows }, inputVector) // Wxh index
    let h1 = graph.mul({ rows: hiddenSize, cols: hiddenSize }, hiddenPrev) // Whh index

    return graph.relu(
      graph.add(graph.add(h0, h1), { rows: hiddenSize, cols: 1, type: 'zeros' }), // bhh index
    )
  }, null)

  // one decoder to outputs at end
  return graph.add(
    graph.mul(
      { rows: outputSize, cols: finalHidden.rows }, // Whd
      finalHidden,
    ),
    { rows: outputSize, cols: 1, type: 'zeros' }, // bd
  )
}

// TODO: Delete this once above is working
// function initRNN(inputSize, hiddenSizes, outputSize) {
//   const model = hiddenSizes.reduce((model, hiddenSize, index, hiddenSizes) => {
//     const prevSize = index === 0 ? inputSize : hiddenSizes[index - 1]

//     model['Wxh' + index] = createRandomMat(hiddenSize, prevSize, 0.08)
//     model['Whh' + index] = createRandomMat(hiddenSize, hiddenSize, 0.08)
//     model['bhh' + index] = createMat(hiddenSize, 1)
//   }, {})

//   // decoder params
//   model['Whd'] = createRandomMat(outputSize, hiddenSizes[hiddenSizes.length - 1], 0.08)
//   model['bd'] = createMat(outputSize, 1)

//   // letter embedding vectors
//   model['Wil'] = createRandomMat(outputSize, inputSize, 0, 0.08)

//   return model
// }
