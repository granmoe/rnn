import { softmax, maxIndex, sampleIndex } from './utils'
import { updateWeights } from './matrix'

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
      updateWeights(output, weight => weight / temperature)
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
