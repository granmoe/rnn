import optimize from './optimize'
import { range } from './utils'

// make a train function that closes around a given graph instance and its params
const makeTrainFunc = ({
  regc,
  clipVal,
  decayRate,
  smoothingEpsilon,
  totalIterations,
  model: { forward, backward, layers, predict },
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

      const { perplexity, cost } = forward()
      backward()
      optimize({
        layers,
        learningRate,
        regc,
        clipVal,
        decayRate,
        smoothingEpsilon,
      })

      if (currentIteration === numIterations) {
        let argMaxPrediction, samples

        if (sampleFrequency && totalIterations % sampleFrequency === 0) {
          argMaxPrediction = predict({
            sample: false,
            textModel,
            temperature,
            maxCharsGen,
          })

          samples = Array.from({ length: 3 }, () =>
            predict({
              sample: true,
              textModel,
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
