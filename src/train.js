import optimize from './optimize'
import { computeCost, predictSentence } from './forward'
import { randInt, range } from './utils'

// make a train function that closes around a given model instance and its params
const makeTrainFunc = ({
  type,
  hiddenSizes,
  regc,
  clipVal,
  decayRate,
  smoothingEpsilon,
  stepCache,
  totalIterations,
  model,
  textModel,
}) => {
  return ({
    numIterations = 1,
    temperature = 1, // how peaky model predictions should be
    learningRate = 0.01,
    maxCharsGen,
  } = {}) => {
    for (const currentIteration of range(1, numIterations)) {
      totalIterations += 1

      const randomSentence = textModel.sentences[randInt(0, textModel.sentences.length)]

      const { graph, perplexity, cost } = computeCost({
        model,
        type,
        textModel,
        hiddenSizes,
        sentence: randomSentence,
      })

      // use built up graph to compute backprop (set .dw fields in mats)
      graph.runBackprop()

      // perform param update using gradients computed during backprop...
      // this all seems kind of indirect...consider restructuring...
      // looks like the purpose of dw is just to keep track of gradients temporarily
      // maybe do this in a functional way...generators?
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
        const argMaxPrediction = predictSentence({
          type,
          model,
          textModel,
          hiddenSizes,
          sample: false,
          temperature,
          maxCharsGen,
        })

        const samples = Array.from({ length: 3 }, () =>
          predictSentence({
            type,
            model,
            textModel,
            hiddenSizes,
            sample: true,
            temperature,
            maxCharsGen,
          }),
        )

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
