import makeTrainFunc from './train'
import Layer, { randLayer, matFromJSON } from './Layer'
import { createLSTM } from './forward'

// TODO: Should create and its child funcs be its own module?
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
  decayRate = 0.994,
  smoothingEpsilon = 1e-8, // to avoid division by zero
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

  const train = makeTrainFunc({
    type,
    hiddenSizes,
    regc,
    clipVal,
    decayRate,
    smoothingEpsilon,
    stepCache,
    totalIterations,
    graph: model,
    textModel,
  })

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
      smoothingEpsilon,
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
    model = createLSTM(letterSize, hiddenSizes, textModel.inputSize)
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

function initRNN(inputSize, hiddenSizes, outputSize) {
  const model = hiddenSizes.reduce((model, hiddenSize, index, hiddenSizes) => {
    const prevSize = index === 0 ? inputSize : hiddenSizes[index - 1]

    model['Wxh' + index] = randLayer(hiddenSize, prevSize, 0.08)
    model['Whh' + index] = randLayer(hiddenSize, hiddenSize, 0.08)
    model['bhh' + index] = new Layer(hiddenSize, 1)
  }, {})

  // decoder params
  model['Whd'] = randLayer(outputSize, hiddenSizes[hiddenSizes.length - 1], 0.08)
  model['bd'] = new Layer(outputSize, 1)

  // letter embedding vectors
  model['Wil'] = randLayer(outputSize, inputSize, 0, 0.08)

  return model
}

// inputSize = letterSize, outputSize = num unique chars
// prettier-ignore
function initLSTM(inputSize, hiddenSizes, outputSize) { // eslint-disable-line
  return hiddenSizes.reduce((model, hiddenSize, index, hiddenSizes) => {
    const prevSize = index === 0 ? inputSize : hiddenSizes[index - 1]

    // input gate
    model['Wix' + index] = randLayer(hiddenSize, prevSize, 0.08)
    model['Wih' + index] = randLayer(hiddenSize, hiddenSize, 0.08)
    model['bi' + index] = new Layer(hiddenSize, 1)
    // forget gate
    model['Wfx' + index] = randLayer(hiddenSize, prevSize, 0.08)
    model['Wfh' + index] = randLayer(hiddenSize, hiddenSize, 0.08)
    model['bf' + index] = new Layer(hiddenSize, 1)
    // output gate
    model['Wox' + index] = randLayer(hiddenSize, prevSize, 0.08)
    model['Woh' + index] = randLayer(hiddenSize, hiddenSize, 0.08)
    model['bo' + index] = new Layer(hiddenSize, 1)

    // cell write params
    model['Wcx' + index] = randLayer(hiddenSize, prevSize, 0.08)
    model['Wch' + index] = randLayer(hiddenSize, hiddenSize, 0.08)
    model['bc' + index] = new Layer(hiddenSize, 1)

    // TODO: Looks like these get overwritten every iteration
    // decoder params
    model['Whd'] = randLayer(outputSize, hiddenSize, 0.08)
    model['bd'] = new Layer(outputSize, 1)

    // letter embedding vectors
    model['Wil'] = randLayer(outputSize, inputSize, 0.08)

    return model
  }, {})
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
