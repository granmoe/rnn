import makeTrainFunc from './train'
import { computeCost, predictSentence } from './forward'
import createGraph from './Graph'
import { randInt } from './utils'

// TODO: Should create and its child funcs be its own module?
export function create({
  // SETS UP THE GRAPH
  modelFunc,
  // BASIC HYPER PARAMS
  type,
  input,
  hiddenSizes = [20, 20],
  letterSize = 5,
  charCountThreshold = 1,
  // OPTIMIZATION HYPER PARAMS
  regc = 0.000001, // L2 regularization strength
  clipVal = 5, // clip gradients at this value
  decayRate = 0.9,
  smoothingEpsilon = 1e-8, // to avoid division by zero
  // these are only passed in when restarting a saved model
  totalIterations = 0,
  models = createModels({
    type,
    input,
    hiddenSizes,
    letterSize,
    charCountThreshold,
    modelFunc,
  }),
}) {
  const { model, textModel } = models

  const train = makeTrainFunc({
    type,
    hiddenSizes,
    regc,
    clipVal,
    decayRate,
    smoothingEpsilon,
    totalIterations,
    model,
    textModel,
  })

  // TODO IO
  // const toJSON = () =>
  //   JSON.stringify({
  //     type,
  //     hiddenSizes,
  //     regc,
  //     clipVal,
  //     totalIterations,
  //     models: {
  //       textModel,
  //       model: modelToJSON(model),
  //     },
  //     smoothingEpsilon,
  //     decayRate,
  //   })

  return {
    train,
    models: {
      model,
      textModel,
    },
    // toJSON,
  }
}

// TODO: This name is kind of awkward
function createModels({ hiddenSizes, letterSize, input, charCountThreshold, modelFunc }) {
  const sentences = input.split('\n').map(str => str.trim())
  const textModel = createTextModel(sentences, charCountThreshold)

  const graph = createGraph()

  const forwardFunc = modelFunc({
    inputSize: letterSize,
    outputSize: textModel.inputSize,
    hiddenSizes,
    graph,
  })

  const computeCostForwardFunc = input => {
    graph.nextLayerIndex = 0
    graph.doBackprop = true
    return forwardFunc(input)
  }

  const predictForwardFunc = input => {
    graph.nextLayerIndex = 0
    graph.doBackprop = false
    return forwardFunc(input)
  }

  const model = {
    forward: () => {
      const randomSentence = textModel.sentences[randInt(0, textModel.sentences.length)]

      return computeCost({
        textModel,
        forward: computeCostForwardFunc,
        sentence: randomSentence,
      })
    },
    // TODO: destructure here for documentation purposes? Or maybe that's what docs are for?
    predict: opts => {
      graph.doBackprop = false
      return predictSentence({
        ...opts,
        forward: predictForwardFunc,
      })
    },
    backward: graph.backward.bind(graph),
    layers: graph.layers,
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

/* TODO IO
function loadFromJSON(json) {
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
    {}
  )
*/
