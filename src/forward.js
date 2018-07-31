import Layer from './Layer'
import { softmax, maxIndex, sampleIndex } from './utils'

export function predictSentence({
  forward,
  textModel,
  maxCharsGen = 100, // length of output
  sample = false,
  temperature = 1,
}) {
  let probs
  let sentence = ''
  let charIndex = 0

  do {
    const output = forward(charIndex) // output used to be called logprobs

    if (temperature !== 1 && sample) {
      // Scale log probabilities by temperature and renormalize
      // If temperature is high, logprobs will go towards zero,
      // and the softmax outputs will be more diffuse. If temperature is
      // very low, the softmax outputs will be more peaky
      output.updateWeights(weight => weight / temperature)
    }
    probs = softmax(output)

    charIndex = sample ? sampleIndex(probs.weights) : maxIndex(probs.weights)

    if (charIndex !== 0) sentence += textModel.indexToLetter[charIndex]
    // 0 index is END token (or is it the beginning of a new sentence?), maxCharsGen is a way to limit the max length of predictions
  } while (charIndex !== 0 && sentence.length <= maxCharsGen)

  return sentence
}

// calculates perplexity and loss of model on a given sentence
export function computeCost({ textModel, sentence, forward }) {
  let log2ppl = 0
  let cost = 0
  const sentenceIndices = Array.from(sentence).map(c => textModel.letterToIndex[c])
  let delimitedSentence = [0, ...sentenceIndices, 0] // start and end tokens are zeros

  // could be: for (let [currentCharIndex, nextCharIndex] of slidingWindow(2, delimitedSentence)) { // except this is really damn slow
  for (let i = 0; i < delimitedSentence.length - 1; i++) {
    const currentCharIndex = delimitedSentence[i]
    const nextCharIndex = delimitedSentence[i + 1]
    const output = forward(currentCharIndex)
    const probs = softmax(output) // compute the softmax probabilities, interpreting output as logprobs

    const nextCharProbability = probs.weights[nextCharIndex]
    // binary logarithm 0 ... 1 = -Infinity ... 1
    log2ppl -= Math.log2(nextCharProbability) // accumulate binary log prob and do smoothing
    // natural logarithm, 0 ... 1 = -Infinity ... 0
    // since softmax will be between 0 and 1 exclusive, cost will always be negative
    // high prob = low cost, low prob = high cost
    cost -= Math.log(nextCharProbability)

    // write gradients into log probabilities
    output.gradients = probs.weights
    output.gradients[nextCharIndex] -= 1
  }

  const perplexity = Math.pow(2, log2ppl / (sentence.length - 1))
  return { perplexity, cost }
}

// TODO: Prob should make the args an object and give descriptive names to all of these...generalize a bit
export const makeForwardLSTM = (inputSize, hiddenSizes, outputSize, graph) => index => {
  const x = graph.rowPluck({ rows: outputSize, cols: inputSize }, index) // Wil

  const finalHidden = hiddenSizes.reduce((prevHidden, hiddenSize, index) => {
    const input = prevHidden || x // output of last layer (but first layer takes input)
    const hiddenPrev = graph.getMat({ rows: hiddenSize, cols: 1, type: 'zeros' }) // 'hiddenPrev' index
    const cellPrev = graph.getMat({ rows: hiddenSize, cols: 1, type: 'zeros' }) // 'cellPrev' index

    // input gate
    const h0 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wix' index
      input,
    )
    const h1 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Wih' index
      hiddenPrev,
    )
    const inputGate = graph.sigmoid(
      graph.add(graph.add(h0, h1), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bi' index
    )

    // forget gate
    const h2 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wfx' index
      input,
    )
    const h3 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Wfh' index
      hiddenPrev,
    )
    const forgetGate = graph.sigmoid(
      graph.add(graph.add(h2, h3), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bf' index
    )

    // output gate
    const h4 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wox' index
      input,
    )
    const h5 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Woh' index
      hiddenPrev,
    )
    const outputGate = graph.sigmoid(
      graph.add(graph.add(h4, h5), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bo' index
    )

    // write operation on cells
    const h6 = graph.mul(
      { rows: hiddenSize, cols: input.rows }, // 'Wcx' index
      input,
    )
    const h7 = graph.mul(
      { rows: hiddenSize, cols: hiddenSize }, // 'Wch' index
      hiddenPrev,
    )
    const cellWrite = graph.tanh(
      graph.add(graph.add(h6, h7), { rows: hiddenSize, cols: 1, type: 'zeros' }), // 'bc' index
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
    hiddenPrevs = hiddenSizes.map(hiddenSize => new Layer(hiddenSize, 1))
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
