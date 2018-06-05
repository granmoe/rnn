import Graph from './Graph'
import { slidingWindow, softmax, maxIndex, samplei } from './utils'
import { forwardRNN, forwardLSTM } from './RNN'

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

    charIndex = sample ? samplei(probs.w) : maxIndex(probs.w)

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
