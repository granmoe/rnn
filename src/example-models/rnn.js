import createLayer from '../layer'

// TODO RNN
// prettier-ignore
function forwardRNN(graph, model, x, prev, hiddenSizes) { // eslint-disable-line
  // forward prop for a single tick of RNN
  // model contains RNN parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden activations from last step

  // TODO: prev.h is the only thing used from prev...just pass in prev.h directly and use default value

  let hiddenPrevs
  if (typeof prev.h === 'undefined') {
    hiddenPrevs = hiddenSizes.map(hiddenSize => createLayer(hiddenSize, 1))
  } else {
    hiddenPrevs = prev.h
  }

  const h = hiddenSizes.reduce((result, hiddenSize, index) => {
    let inputVector = index === 0 ? x : result[index - 1]
    let hiddenPrev = hiddenPrevs[index]

    let h0 = graph.mul(model['Wxh' + index], inputVector)
    let h1 = graph.mul(model['Whh' + index], hiddenPrev)

    return graph.relu(graph.add(graph.add(h0, h1), model['bhh' + index]))
  }, [])

  // one decoder to outputs at end
  const o = graph.add(graph.mul(model['Whd'], h[h.length - 1]), model['bd'])

  // return cell memory, hidden representation and output
  return { h, o }
}
