import optimize from './optimize'
import { predictSentence } from './forward'
import { softmax, randInt, range } from './utils'

// make a train function that closes around a given graph instance and its params
const makeTrainFunc = ({
  type,
  hiddenSizes,
  regc,
  clipVal,
  decayRate,
  smoothingEpsilon,
  totalIterations,
  graph,
  textModel,
}) => {
  return ({
    numIterations = 1,
    temperature = 1, // how peaky model predictions should be
    learningRate = 0.01,
    maxCharsGen,
    sampleFrequency = null, // how often to return samples and argmax (don't sample if not a number > 1)
  } = {}) => {
    for (const currentIteration of range(1, numIterations)) {
      totalIterations += 1

      const randomSentence = textModel.sentences[randInt(0, textModel.sentences.length)]

      const { perplexity, cost } = train({
        graph,
        regc,
        clipVal,
        decayRate,
        smoothingEpsilon,
        learningRate,
        type,
        textModel,
        sentence: randomSentence,
      })

      if (currentIteration === numIterations) {
        let argMaxPrediction, samples
        if (sampleFrequency && totalIterations % sampleFrequency === 0) {
          argMaxPrediction = predictSentence({
            type,
            graph,
            textModel,
            hiddenSizes,
            sample: false,
            temperature,
            maxCharsGen,
          })

          samples = Array.from({ length: 3 }, () =>
            predictSentence({
              type,
              graph,
              textModel,
              hiddenSizes,
              sample: true,
              temperature,
              maxCharsGen,
            }),
          )
        }

        return {
          iterations: totalIterations,
          perplexity,
          cost,
          argMaxPrediction,
          samples,
        }
      }
    }
  }
}

export default makeTrainFunc

// Runs forward, backward, and optimize char by char for a full sample (sentence)
export function train({
  textModel,
  sentence,
  graph,
  learningRate,
  regc,
  clipVal,
  decayRate,
  smoothingEpsilon,
}) {
  let log2ppl = 0
  let cost = 0
  const sentenceIndices = Array.from(sentence).map(c => textModel.letterToIndex[c])
  let delimitedSentence = [0, ...sentenceIndices, 0] // start and end tokens are zeros

  for (let i = 0; i < delimitedSentence.length - 1; i++) {
    const currentCharIndex = delimitedSentence[i]
    const nextCharIndex = delimitedSentence[i + 1]
    const output = graph.forward(currentCharIndex)
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

    graph.backward()

    optimize({
      graph,
      learningRate,
      regc,
      clipVal,
      decayRate,
      smoothingEpsilon,
    })
  }

  const perplexity = Math.pow(2, log2ppl / (sentence.length - 1))
  return { perplexity, cost }
}
