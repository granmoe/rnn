// import optimize from './optimize'
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

      const { perplexity, cost } = computeCost({
        graph,
        type,
        textModel,
        sentence: randomSentence,
      })

      // use built up graph to compute backprop (set .gradients fields in mats)
      graph.backward()

      // perform param update using gradients computed during backprop...
      // this all seems kind of indirect...consider restructuring...
      // looks like the purpose of gradient is just to keep track of gradients temporarily
      // maybe do this in a functional way...generators?
      // TODO: Get this working
      // optimize({
      //   graph,
      //   learningRate,
      //   regc,
      //   clipVal,
      //   decayRate,
      //   smoothingEpsilon,
      //   stepCache,
      // })

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
