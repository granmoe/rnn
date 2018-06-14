import Graph from './Graph'
import Mat from './Mat'
import { slidingWindow, softmax, maxIndex, sampleIndex } from './utils'

// calculates loss of model on a given sentence and returns graph to be used for backprop
// graph.runBackprop side-effects model via closures
export function computeCost({ type, model, textModel, hiddenSizes, sentence }) {
  const graph = new Graph()
  let log2ppl = 0
  let cost = 0
  let lh = {}
  const sentenceIndices = Array.from(sentence).map(c => textModel.letterToIndex[c])
  let delimitedSentence = [0, ...sentenceIndices, 0] // start and end tokens are zeros

  for (let [currentCharIndex, nextCharIndex] of slidingWindow(2, delimitedSentence)) {
    // TODO: Why "lh?" Change this...expand out to whatever the acronym stands for if possible
    lh = forwardIndex(graph, model, currentCharIndex, lh, hiddenSizes, type)
    const probs = softmax(lh.o) // compute the softmax probabilities, interpreting output as logprobs

    const nextCharProbability = probs.w[nextCharIndex]
    // binary logarithm 0 ... 1 = -Infinity ... 1
    log2ppl -= Math.log2(nextCharProbability) // accumulate binary log prob and do smoothing
    // natural logarithm, 0 ... 1 = -Infinity ... 0
    // since softmax will be between 0 and 1 exclusive, cost will always be negative
    // high prob = low cost, low prob = high cost
    cost -= Math.log(nextCharProbability)

    // write gradients into log probabilities
    lh.o.dw = probs.w
    lh.o.dw[nextCharIndex] -= 1
  }

  const perplexity = Math.pow(2, log2ppl / (sentence.length - 1))
  return { graph, perplexity, cost }
}

export function predictSentence({
  type,
  model,
  textModel,
  hiddenSizes,
  maxCharsGen = 100, // length of output
  sample = false,
  temperature = 1,
}) {
  let lh, logprobs, probs
  let graph = new Graph({ doBackprop: false }) // Just predict (forward), don't do backprop
  let sentence = ''
  let prev = {}
  let charIndex = 0

  do {
    lh = forwardIndex(graph, model, charIndex, prev, hiddenSizes, type)
    prev = lh

    logprobs = lh.o
    if (temperature !== 1 && sample) {
      // Scale log probabilities by temperature and renormalize
      // If temperature is high, logprobs will go towards zero,
      // and the softmax outputs will be more diffuse. If temperature is
      // very low, the softmax outputs will be more peaky
      logprobs.updateW(w => w / temperature)
    }
    probs = softmax(logprobs)

    charIndex = sample ? sampleIndex(probs.w) : maxIndex(probs.w)

    if (charIndex !== 0) sentence += textModel.indexToLetter[charIndex]
    // 0 index is END token (or is it the beginning of a new sentence?), maxCharsGen is a way to limit the max length of predictions
  } while (charIndex !== 0 && sentence.length <= maxCharsGen)

  return sentence
}

function forwardIndex(G, model, ix, prev, hiddenSizes, type) {
  // Could this somehow be how prev.o.dw is having an effect?
  const x = G.rowPluck(model['Wil'], ix) // char embedding for given char
  // forward prop the sequence learner
  return type === 'rnn'
    ? forwardRNN(G, model, x, prev, hiddenSizes)
    : forwardLSTM(G, model, x, prev, hiddenSizes)
}

// TODO: further refactoring here and make sure to understand everything
function forwardLSTM(graph, model, x, prev, hiddenSizes) {
  // forward prop for a single tick of LSTM
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
      let h0 = graph.mul(model['Wix' + index], inputVector)
      let h1 = graph.mul(model['Wih' + index], hiddenPrev)
      let inputGate = graph.sigmoid(graph.add(graph.add(h0, h1), model['bi' + index]))

      // forget gate
      let h2 = graph.mul(model['Wfx' + index], inputVector)
      let h3 = graph.mul(model['Wfh' + index], hiddenPrev)
      let forgetGate = graph.sigmoid(graph.add(graph.add(h2, h3), model['bf' + index]))

      // output gate
      let h4 = graph.mul(model['Wox' + index], inputVector)
      let h5 = graph.mul(model['Woh' + index], hiddenPrev)
      let outputGate = graph.sigmoid(graph.add(graph.add(h4, h5), model['bo' + index]))

      // write operation on cells
      let h6 = graph.mul(model['Wcx' + index], inputVector)
      let h7 = graph.mul(model['Wch' + index], hiddenPrev)
      let cellWrite = graph.tanh(graph.add(graph.add(h6, h7), model['bc' + index]))

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
  let output = graph.add(graph.mul(model['Whd'], hidden[hidden.length - 1]), model['bd'])

  // return cell memory, hidden representation and output
  return { h: hidden, c: cell, o: output }
}

function forwardRNN(graph, model, x, prev, hiddenSizes) {
  // forward prop for a single tick of RNN
  // model contains RNN parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden activations from last step

  // TODO: prev.h is the only thing used from prev...just pass in prev.h directly and use default value

  let hiddenPrevs
  if (typeof prev.h === 'undefined') {
    hiddenPrevs = hiddenSizes.map(hiddenSize => new Mat(hiddenSize, 1))
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
