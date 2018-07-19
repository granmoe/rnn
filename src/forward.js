import Graph from './Graph'
import Mat, { randMat } from './Mat'
import { softmax, maxIndex, sampleIndex } from './utils'

export function predictSentence({
  graph,
  textModel,
  maxCharsGen = 100, // length of output
  sample = false,
  temperature = 1,
}) {
  let logprobs, probs
  let sentence = ''
  let charIndex = 0

  do {
    const output = graph.forward(charIndex, { doBackprop: false })

    logprobs = output
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

// calculates loss of model on a given sentence and returns graph to be used for backprop
// graph.runBackprop side-effects model via closures
export function computeCost({ textModel, sentence, graph, type }) {
  let log2ppl = 0
  let cost = 0
  const sentenceIndices = Array.from(sentence).map(c => textModel.letterToIndex[c])
  let delimitedSentence = [0, ...sentenceIndices, 0] // start and end tokens are zeros

  // TODO: Maybe someday I can change this back to my pretty, customer iterator :(
  // But for now, for...of is just too damn slow
  // for (let [currentCharIndex, nextCharIndex] of slidingWindow(2, delimitedSentence)) {
  for (let i = 0; i < delimitedSentence.length - 1; i++) {
    const currentCharIndex = delimitedSentence[i]
    const nextCharIndex = delimitedSentence[i + 1]
    // TODO: Why "lh?" Change this...expand out to whatever the acronym stands for if possible
    const output = graph.forward(currentCharIndex, { doBackprop: true }) // TODO: Turn this back on eventually
    // lh = forwardIndex(graph, model, currentCharIndex, lh, hiddenSizes, type)
    const probs = softmax(output) // compute the softmax probabilities, interpreting output as logprobs

    const nextCharProbability = probs.w[nextCharIndex]
    // binary logarithm 0 ... 1 = -Infinity ... 1
    log2ppl -= Math.log2(nextCharProbability) // accumulate binary log prob and do smoothing
    // natural logarithm, 0 ... 1 = -Infinity ... 0
    // since softmax will be between 0 and 1 exclusive, cost will always be negative
    // high prob = low cost, low prob = high cost
    cost -= Math.log(nextCharProbability)

    // write gradients into log probabilities
    output.dw = probs.w
    output.dw[nextCharIndex] -= 1
  }

  /*
    TODO BI-RNN: 
    
    Sum the gradients of backward/forward
    Sum the log2ppl and cost

    forward/backward both create 3 arrays: grads, log2ppl and cost
  */

  const perplexity = Math.pow(2, log2ppl / (sentence.length - 1))
  return { graph, perplexity, cost }
}

// TODO: Grab declarations of mats from create model and merge them into here
// Won't need to pass any args into here...except maybe hiddenSizes (are those on model?)
export function createLSTM(inputSize, hiddenSizes, outputSize) {
  // forward prop for a single tick of LSTM, model contains LSTM parameters
  // x is 1D column vector with observation
  const graph = new Graph()

  const x = graph.rowPluck(randMat(outputSize, inputSize)) // Wil, graph.rowPluck then waits for second arg, which is the row vector of the letter at the given index

  // could maybe create these in the reduce below?
  const hiddenPrevs = hiddenSizes.map(hiddenSize => new Mat(hiddenSize, 1))
  const cellPrevs = [...hiddenPrevs]

  const finalHidden = hiddenSizes.reduce((prevHidden, hiddenSize, index) => {
    const input = prevHidden || x // output of last layer (but first layer takes input)
    const hiddenPrev = hiddenPrevs[index]
    const cellPrev = cellPrevs[index]

    // input gate
    const h0 = graph.mul(randMat(hiddenSize, input.rows), input) // randMat is Wix[index] / layer['Wix']
    const h1 = graph.mul(randMat(hiddenSize, hiddenSize), hiddenPrev) // randMat is Wih[index]
    const inputGate = graph.sigmoid(graph.add(graph.add(h0, h1), new Mat(hiddenSize, 1))) // currentLayer.bi

    // forget gate
    const h2 = graph.mul(randMat(hiddenSize, input.rows), input) // Wfx
    const h3 = graph.mul(randMat(hiddenSize, hiddenSize), hiddenPrev) // Wfh
    const forgetGate = graph.sigmoid(graph.add(graph.add(h2, h3), new Mat(hiddenSize, 1))) // bf

    // output gate
    const h4 = graph.mul(randMat(hiddenSize, input.rows), input) // Wox
    const h5 = graph.mul(randMat(hiddenSize, hiddenSize), hiddenPrev) // Woh
    const outputGate = graph.sigmoid(graph.add(graph.add(h4, h5), new Mat(hiddenSize, 1))) // bo

    // write operation on cells
    const h6 = graph.mul(randMat(hiddenSize, input.rows), input) // Wcx
    const h7 = graph.mul(randMat(hiddenSize, hiddenSize), hiddenPrev) // Wch
    const cellWrite = graph.tanh(graph.add(graph.add(h6, h7), new Mat(hiddenSize, 1))) // bc

    // compute new cell activation
    const retainCell = graph.eltmul(forgetGate, cellPrev) // what do we keep from cell
    const writeCell = graph.eltmul(inputGate, cellWrite) // what do we write to cell
    const cellD = graph.add(retainCell, writeCell) // new cell contents

    // compute hidden state as gated, saturated cell activations and pass it to next iteration
    return graph.eltmul(outputGate, graph.tanh(cellD))
  }, null)

  // output, one decoder to outputs at end
  graph.add(
    graph.mul(randMat(outputSize, finalHidden.rows), finalHidden), // Whd
    new Mat(outputSize, 1), // bd
  )

  // return the built up graph, which has a forward function we will use to do forward prop
  return graph
}

// TODO RNN
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