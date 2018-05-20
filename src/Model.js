import Solver from './Solver'
import Graph from './Graph'
import { repeat, randi, softmax, maxi, samplei } from './utils'
import { initRNN, initLSTM, forwardRNN, forwardLSTM } from './RNN'
import { matFromJson } from './Mat'

export function loadFromJson(json) {
  const args = JSON.parse(json)

  return create({
    ...args,
    models: {
      model: modelFromJson(args.models.model),
      textModel: args.models.textModel,
    },
  })
}

// returns a function that will train the model
export function create({
  // BASIC HYPER PARAMS
  type,
  input,
  hiddenSizes = [20, 20],
  letterSize = 5,
  charCountThreshold = 1,
  // OPTIMIZATION HYPER PARAMS
  regc = 0.000001, // L2 regularization strength
  learningRate = 0.01,
  clipVal = 5, // clip gradients at this value
  // PREDICTION HYPER PARAMS
  temperature = 1, // how peaky model predictions should be
  maxCharsGen = 100, // max length of generated sentences
  models = createModels({
    type,
    input,
    hiddenSizes,
    letterSize,
    charCountThreshold,
  }),
}) {
  const { model, textModel } = models

  let solver = new Solver()
  let totalIterations = 1

  const train = ({ iterations = 1, temperature = 1 } = {}) => {
    let currentIteration = 1
    let result

    repeat(iterations, () => {
      // sample sentence from data
      const sentence = textModel.sentences[randi(0, textModel.sentences.length)]
      // evaluate cost function on a sentence
      const cost = costFunc({
        model,
        textModel,
        hiddenSizes,
        sentence,
      })
      // use built up graph to compute backprop (set .dw fields in mats)
      cost.G.runBackprop()
      // perform param update
      solver.step(model, learningRate, regc, clipVal)

      const argMaxPrediction = predictSentence({
        model,
        textModel,
        hiddenSizes,
        maxCharsGen,
        sample: false,
        temperature,
      })

      let samples = []
      repeat(5, () => {
        samples.push(
          predictSentence({
            model,
            textModel,
            hiddenSizes,
            maxCharsGen,
            sample: true,
            temperature,
          }),
        )
      })

      if (currentIteration === iterations) {
        result = { argMaxPrediction, samples, iterations: totalIterations }
      }

      currentIteration += 1
      totalIterations += 1
    })

    return result
  }

  const toJSON = () =>
    JSON.stringify({
      type,
      hiddenSizes,
      letterSize,
      charCountThreshold,
      regc,
      learningRate,
      clipVal,
      temperature,
      maxCharsGen,
      models: {
        textModel,
        model: modelToJson(model),
      },
    })

  return {
    train,
    toJSON,
  }
}

function forwardIndex(G, model, ix, prev, hiddenSizes) {
  // TODO: Should just be a method on the model
  // Then no need for branching based on h params and no need for so much indirection
  const x = G.rowPluck(model['Wil'], ix)
  // forward prop the sequence learner
  return hiddenSizes.type === 'rnn'
    ? forwardRNN(G, model, x, prev, hiddenSizes)
    : forwardLSTM(G, model, x, prev, hiddenSizes)
}

function predictSentence({
  model,
  textModel,
  hiddenSizes,
  maxCharsGen,
  sample = false,
  temperature = 1,
}) {
  let lh, logprobs, probs
  let G = new Graph(false)
  let s = ''
  let prev = {}
  while (true) {
    // RNN tick
    let ix = s.length === 0 ? 0 : textModel.letterToIndex[s[s.length - 1]]
    lh = forwardIndex(G, model, ix, prev, hiddenSizes)
    prev = lh

    // sample predicted letter
    logprobs = lh.o
    if (temperature !== 1 && sample) {
      // scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky
      for (let q = 0, nq = logprobs.w.length; q < nq; q++) {
        logprobs.w[q] /= temperature
      }
    }

    probs = softmax(logprobs)
    if (sample) {
      ix = samplei(probs.w)
    } else {
      ix = maxi(probs.w)
    }

    if (ix === 0) break // END token predicted, break out
    if (s.length > maxCharsGen) break // something is wrong

    let letter = textModel.indexToLetter[ix]
    s += letter
  }

  return s
}

// TODO
// side-effects: model, lh, logprobs, probs
function costFunc({ model, textModel, hiddenSizes, sentence }) {
  // takes a model and a sentence and
  // calculates the loss. Also returns the Graph
  // object which can be used to do backprop
  let lh, logprobs, probs
  let n = sentence.length
  let G = new Graph()
  let log2ppl = 0.0
  let cost = 0.0
  let prev = {}
  for (let i = -1; i < n; i++) {
    // start and end tokens are zeros
    let ixSource = i === -1 ? 0 : textModel.letterToIndex[sentence[i]] // first step: start with START token
    let ixTarget = i === n - 1 ? 0 : textModel.letterToIndex[sentence[i + 1]] // last step: end with END token

    lh = forwardIndex(G, model, ixSource, prev, hiddenSizes)
    prev = lh

    // set gradients into logprobabilities
    logprobs = lh.o // interpret output as logprobs
    probs = softmax(logprobs) // compute the softmax probabilities

    log2ppl += -Math.log2(probs.w[ixTarget]) // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ixTarget])

    // write gradients into log probabilities
    logprobs.dw = probs.w
    logprobs.dw[ixTarget] -= 1
  }
  const ppl = Math.pow(2, log2ppl / (n - 1))
  return { G, ppl, cost }
}

// TODO: This name is kind of awkward
function createModels({
  type,
  hiddenSizes,
  letterSize,
  input,
  charCountThreshold,
}) {
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

const modelToJson = model =>
  Object.entries(model).reduce(
    (result, [matName, mat]) => ({
      ...result,
      [matName]: mat.serialize(),
    }),
    {},
  )

const modelFromJson = model =>
  Object.entries(model).reduce(
    (result, [matName, matJson]) => ({
      ...result,
      [matName]: matFromJson(matJson),
    }),
    {},
  )
