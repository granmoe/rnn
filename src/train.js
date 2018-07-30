import optimize from './optimize'
import { computeCost, predictSentence } from './forward'
import { randInt, range } from './utils'

// make a train function that closes around a given graph instance and its params
const makeTrainFunc = ({
  type,
  hiddenSizes,
  regc,
  clipVal,
  decayRate,
  smoothingEpsilon,
  stepCache,
  totalIterations,
  graph: forwardLSTM,
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

      const { perplexity, cost, graph, model } = computeCost({
        forwardLSTM,
        // type,
        textModel,
        sentence: randomSentence,
      })

      graph.backward()

      optimize({
        model,
        learningRate,
        regc,
        clipVal,
        decayRate,
        smoothingEpsilon,
        stepCache,
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
