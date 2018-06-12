import optimize from './optimize'
import { computeCost, predictSentence } from './forward'
import { randInt, range } from './utils'
import { initRNN, initLSTM } from './RNN'
import { matFromJSON } from './Mat'

export function create({
  // BASIC HYPER PARAMS
  type,
  input,
  hiddenSizes = [20, 20],
  letterSize = 5,
  charCountThreshold = 1,
  // OPTIMIZATION HYPER PARAMS
  regc = 0.000001, // L2 regularization strength
  clipVal = 5, // clip gradients at this value
  decayRate = 0.999,
  smoothEps = 1e-8,
  // these are only passed in when restarting a saved model
  stepCache = {},
  totalIterations = 0,
  models = createModels({
    type,
    input,
    hiddenSizes,
    letterSize,
    charCountThreshold,
  }),
}) {
  const { model, textModel } = models

  const train = ({
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
      // maybe do this in a functional way
      optimize({
        model,
        learningRate,
        regc,
        clipVal,
        decayRate,
        smoothEps,
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

  const toJSON = () =>
    JSON.stringify({
      type,
      hiddenSizes,
      regc,
      clipVal,
      totalIterations,
      models: {
        textModel,
        model: modelToJSON(model),
      },
      smoothEps,
      decayRate,
      stepCache: modelToJSON(stepCache),
    })

  return {
    train,
    toJSON,
    models: {
      model,
      textModel,
    },
    hiddenSizes,
  }
}

// TODO: This name is kind of awkward
function createModels({ type, hiddenSizes, letterSize, input, charCountThreshold }) {
  const sentences = input.split('\n').map(str => str.trim())
  const textModel = createTextModel(sentences, charCountThreshold)

  let model
  if (type === 'rnn') {
    model = initRNN(letterSize, hiddenSizes, textModel.inputSize)
  } else {
    model = initLSTM(letterSize, hiddenSizes, textModel.inputSize)
  }

  return { model, textModel }
}

function createTextModel(sentences, charCountThreshold = 1) {
  // go over all characters and keep track of all unique ones seen
  const charCounts = [...sentences.join('')].reduce((counts, char) => {
    counts[char] = counts[char] ? counts[char] + 1 : (counts[char] = 1)
    return counts
  }, {})

  // Note: inputSize is one greater than charList.length due to START and END tokens
  // START token will be index 0 in model letter vectors
  // END token will be index 0 in the next character softmax
  const initialVocabData = {
    sentences,
    letterToIndex: {},
    indexToLetter: {},
    charList: [],
    inputSize: 1,
  }

  return Object.entries(charCounts).reduce((result, [char, count]) => {
    if (count >= charCountThreshold) {
      result.charList.push(char)
      result.letterToIndex[char] = result.charList.length
      result.indexToLetter[result.charList.length] = char
      result.inputSize += 1
    }

    return result
  }, initialVocabData)
}

export function loadFromJSON(json) {
  const args = JSON.parse(json)

  return create({
    ...args,
    stepCache: modelFromJSON(args.stepCache),
    models: {
      model: modelFromJSON(args.models.model),
      textModel: args.models.textModel,
    },
  })
}

const modelToJSON = model =>
  Object.entries(model).reduce(
    (result, [matName, mat]) => ({
      ...result,
      [matName]: mat.serialize(),
    }),
    {},
  )

const modelFromJSON = model =>
  Object.entries(model).reduce(
    (result, [matName, matJSON]) => ({
      ...result,
      [matName]: matFromJSON(matJSON),
    }),
    {},
  )
