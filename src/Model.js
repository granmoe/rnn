import optimize from './optimize'
import Graph from './Graph'
import { repeat, randi, softmax, maxi, samplei } from './utils'
import { initRNN, initLSTM, forwardRNN, forwardLSTM } from './RNN'
import { matFromJSON } from './Mat'

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
  clipVal = 5, // clip gradients at this value
  totalIterations = 0,
  stepCache = {},
  decayRate = 0.999,
  smoothEps = 1e-8,
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
      // TRAIN MODEL (DO FORWARD / BACKWARD PROP AND REGULARIZE AND CLIP GRADIENTS)
      // get random sentence from data
      const sentence = textModel.sentences[randi(0, textModel.sentences.length)]
      // evaluate cost function on sentence
      const cost = costFunc({
        model,
        textModel,
        hiddenSizes,
        sentence,
      })
      // use built up graph to compute backprop (set .dw fields in mats)
      cost.G.runBackprop()
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
          model,
          textModel,
          hiddenSizes,
          sample: false,
          temperature,
          maxCharsGen,
        })

        const samples = Array.from({ length: 3 }, () =>
          predictSentence({
            model,
            maxCharsGen,
            textModel,
            hiddenSizes,
            sample: true,
            temperature,
          }),
        )

        return {
          iterations: totalIterations,
          perplexity: cost.ppl,
          cost: cost.cost,
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

function forwardIndex(G, model, ix, prev, hiddenSizes) {
  const x = G.rowPluck(model['Wil'], ix)
  // forward prop the sequence learner
  return hiddenSizes.type === 'rnn'
    ? forwardRNN(G, model, x, prev, hiddenSizes)
    : forwardLSTM(G, model, x, prev, hiddenSizes)
}

export function predictSentence({
  model,
  textModel,
  hiddenSizes,
  maxCharsGen = 100, // length of output
  sample = false,
  temperature = 1,
}) {
  let lh, logprobs, probs
  let G = new Graph({ doBackprop: false }) // Just predict (forward), don't do backprop
  let sentence = ''
  let prev = {}
  let charIndex = 0

  do {
    lh = forwardIndex(G, model, charIndex, prev, hiddenSizes)
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
    // 0 index is END token, maxCharsGen is a way to limit the max length of predictions
  } while (charIndex !== 0 && sentence.length <= maxCharsGen)

  return sentence
}

// TODO: side-effects model? maybe not
export function costFunc({ model, textModel, hiddenSizes, sentence }) {
  // Takes a model and a sentence and calculates the loss.
  // Also returns the Graph object used to do backprop
  let lh, probs
  let n = sentence.length
  let G = new Graph()
  let log2ppl = 0.0
  let cost = 0.0
  let prev = {}

  for (let i = -1; i < n; i++) {
    // start and end tokens are zeros
    let ixSource = i === -1 ? 0 : textModel.letterToIndex[sentence[i]] // first step: start with START token
    let ixTarget = i === n - 1 ? 0 : textModel.letterToIndex[sentence[i + 1]] // last step: end with END token

    lh = forwardIndex(G, model, ixSource, prev, hiddenSizes) // TODO: Side-effects model?

    probs = softmax(lh.o) // compute the softmax probabilities, interpreting output as logprobs

    log2ppl += -Math.log2(probs.w[ixTarget]) // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ixTarget])

    // write gradients into log probabilities (this might not do anything)
    lh.o.dw = probs.w
    lh.o.dw[ixTarget] -= 1
    prev = lh
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
  const tokens = tokenize({ input: sentences, level: 'char' })
  const charCounts = tokens.reduce((counts, char) => {
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

const tokenize = ({ input, level = 'char' }) => {
  if (level === 'char') {
    return [...input.join('')]
  } else if (level === 'word') {
    return input.join(' ').split(/\b/)
  }
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
