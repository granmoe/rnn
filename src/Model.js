import optimize from './optimize'
import Graph from './Graph'
import {
  repeat,
  bidirectionalSlidingWindow,
  slidingWindow,
  randi,
  softmax,
  maxi,
  samplei,
} from './utils'
import { initRNN, initLSTM, forwardRNN, forwardLSTM } from './RNN'
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
  } = {}) =>
    repeat(numIterations, currentIteration => {
      totalIterations += 1

      const randomSentence = textModel.sentences[randi(0, textModel.sentences.length)]

      const { graph, graphR, perplexity, cost } = computeCost({
        model,
        type,
        textModel,
        hiddenSizes,
        sentence: randomSentence,
      })
      // use built up graph to compute backprop (set .dw fields in mats)
      graph.runBackprop()
      graphR.runBackprop()
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

function forwardIndex(G, model, ix, prev, hiddenSizes, type) {
  // Could this somehow be how prev.o.dw is having an effect?
  const x = G.rowPluck(model['Wil'], ix) // char embedding for given char
  // forward prop the sequence learner
  return type === 'rnn'
    ? forwardRNN(G, model, x, prev, hiddenSizes)
    : forwardLSTM(G, model, x, prev, hiddenSizes)
}

export function predictSentence({
  type,
  model,
  textModel,
  hiddenSizes,
  maxCharsGen = 100, // length of output
  sample = false,
  temperature = 1,
}) {
  let lh, logprobs, probs
  let graph = new Graph({ doBackprop: false }) // Just predict (forward), don't do backprop
  let sentence = ''
  let prev = {}
  let charIndex = 0

  do {
    lh = forwardIndex(graph, model, charIndex, prev, hiddenSizes, type)
    prev = lh

    logprobs = lh.o
    if (temperature !== 1 && sample) {
      // Scale log probabilities by temperature and renormalize
      // If temperature is high, logprobs will go towards zero,
      // and the softmax outputs will be more diffuse. If temperature is
      // very low, the softmax outputs will be more peaky
      logprobs.updateW(w => w / temperature)
    }
    probs = softmax(logprobs)

    charIndex = sample ? samplei(probs.w) : maxi(probs.w)

    if (charIndex !== 0) sentence += textModel.indexToLetter[charIndex]
    // 0 index is END token (or is it the beginning of a new sentence?), maxCharsGen is a way to limit the max length of predictions
  } while (charIndex !== 0 && sentence.length <= maxCharsGen)

  return sentence
}

// calculates loss of model on a given sentence and returns graph to be used for backprop
export function computeCost({ type, model, textModel, hiddenSizes, sentence }) {
  // TODO: Need to do something different with graph and model in lhR forwardIndex
  // perhaps need separate graph and model?
  const graph = new Graph()
  const graphR = new Graph()
  let log2ppl = 0
  let cost = 0
  let lh = {}
  let lhR = {}
  const sentenceIndices = Array.from(sentence).map(c => textModel.letterToIndex[c])
  let delimitedSentence = [0, ...sentenceIndices, 0] // start and end tokens are zeros

  for (let [currIx, nextIx, currReverseIx, nextReverseIx] of bidirectionalSlidingWindow(
    2,
    delimitedSentence,
  )) {
    // for (let [currentCharIndex, nextCharIndex] of slidingWindow(2, delimitedSentence)) {
    // TODO: Why "lh?" Change this...expand out to whatever the acronym stands for if possible
    lh = forwardIndex(graph, model, currIx, lh, hiddenSizes, type)
    lhR = forwardIndex(graphR, model, currReverseIx, lhR, hiddenSizes, type)
    const probs = softmax(lh.o) // compute the softmax probabilities, interpreting output as logprobs
    const probsR = softmax(lhR.o) // same for reversed

    // accumulate binary log prob and do smoothing
    log2ppl += -((Math.log2(probs.w[nextIx]) + Math.log2(probsR.w[nextReverseIx])) / 2)
    cost += -((Math.log(probs.w[nextIx]) + Math.log(probsR.w[nextReverseIx])) / 2)

    // write gradients into log probabilities
    lh.o.dw = probs.w
    lh.o.dw[nextIx] -= 1
    lhR.o.dw = probsR.w
    lhR.o.dw[nextReverseIx] -= 1
  }

  const perplexity = Math.pow(2, log2ppl / (sentence.length - 1))
  return { graph, graphR, perplexity, cost }
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
