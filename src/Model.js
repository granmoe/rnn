import Solver from './Solver'
import Graph from './Graph'
import RandMat from './Mat'
import { randi, softmax, maxi, samplei } from './utils'
import { initRNN, initLSTM, forwardRNN, forwardLSTM } from './RNN'

let pplList = []
let tickIter = 0

// model parameters
const generator = 'lstm' // can be 'rnn' or 'lstm'
const hiddenSizes = [20, 20] // list of sizes of hidden layers
const letterSize = 5 // size of letter embeddings

// optimization
const regc = 0.000001 // L2 regularization strength
const learningRate = 0.01 // learning rate
const clipval = 5.0 // clip gradients at this value
// prediction params
let sampleSoftmaxTemperature = 1.0 // how peaky model predictions should be
let maxCharsGen = 100 // max length of generated sentences

// various global var inits
// let epochSize = -1
let inputSize = -1
let outputSize = -1
let letterToIndex = {}
let indexToLetter = {}
let vocab = []
let dataSents = []
let solver = new Solver() // should be class because it needs memory for step caches
// let pplGraph = new Rvis()

let lh, logprobs, probs

let model = {}

function initVocab(sents, countThreshold) {
  // go over all characters and keep track of all unique ones seen
  const charCounts = Array.from(sents.join('')).reduce((counts, char) => {
    counts[char] = counts[char] ? counts[char] + 1 : (counts[char] = 1)
    return counts
  }, {})

  // NOTE: start at nextIndex at 1 because we will have START and END tokens!
  // that is, START token will be index 0 in model letter vectors
  // and END token will be index 0 in the next character softmax
  const { letterToIndex: l, indexToLetter: i, vocab: v } = Object.entries(
    charCounts,
  ).reduce(
    (result, [char, count]) => {
      if (count >= countThreshold) {
        result.vocab.push(char)
        result.letterToIndex[char] = result.nextIndex
        result.indexToLetter[result.nextIndex] = char
        result.nextIndex += 1
      }
      return result
    },
    {
      letterToIndex: {},
      indexToLetter: {},
      vocab: [],
      nextIndex: 1,
    },
  )

  letterToIndex = l
  indexToLetter = i
  vocab = v

  // globals written: indexToLetter, letterToIndex, vocab (list), and:
  inputSize = vocab.length + 1
  outputSize = vocab.length + 1
  // epochSize = sents.length
  // TODO: Show this in the UI
  // $('#prepro_status').text(
  // 'found ' + vocab.length + ' distinct characters: ' + vocab.join(''),
  // )
}

function initModel() {
  // letter embedding vectors
  let model = {}
  model['Wil'] = new RandMat(inputSize, letterSize, 0, 0.08)

  if (generator === 'rnn') {
    let rnn = initRNN(letterSize, hiddenSizes, outputSize)
    model = {
      ...model,
      ...rnn,
    }
  } else {
    let lstm = initLSTM(letterSize, hiddenSizes, outputSize)
    model = {
      ...model,
      ...lstm,
    }
  }

  return model
}

export function reinit(inputSentences) {
  // note: reinit writes global vars by running
  // eval on a textarea
  // TODO: Allow user to set hyperparams in a safer way, via inputs

  solver = new Solver() // GLOBAL
  // pplGraph = new Rvis() // GLOBAL

  pplList = [] // GLOBAL
  tickIter = 0 // GLOBAL

  dataSents = inputSentences.split('\n').map(str => str.trim())
  initVocab(dataSents, 1) // takes count threshold for characters
  model = initModel() // pass in some of the stuff that will be returned from initVocab
}

function forwardIndex(G, model, ix, prev) {
  const x = G.rowPluck(model['Wil'], ix)
  // forward prop the sequence learner
  return generator === 'rnn'
    ? forwardRNN(G, model, hiddenSizes, x, prev)
    : forwardLSTM(G, model, hiddenSizes, x, prev)
}

function predictSentence(model, sample = false, temperature = 1.0) {
  let G = new Graph(false)
  let s = ''
  let prev = {}
  while (true) {
    // RNN tick
    let ix = s.length === 0 ? 0 : letterToIndex[s[s.length - 1]]
    lh = forwardIndex(G, model, ix, prev)
    prev = lh

    // sample predicted letter
    logprobs = lh.o
    if (temperature !== 1.0 && sample) {
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
    if (s.length > maxCharsGen) {
      break
    } // something is wrong

    let letter = indexToLetter[ix]
    s += letter
  }
  return s
}

function costfun(model, sent) {
  // takes a model and a sentence and
  // calculates the loss. Also returns the Graph
  // object which can be used to do backprop
  let n = sent.length
  let G = new Graph()
  let log2ppl = 0.0
  let cost = 0.0
  let prev = {}
  for (let i = -1; i < n; i++) {
    // start and end tokens are zeros
    let ixSource = i === -1 ? 0 : letterToIndex[sent[i]] // first step: start with START token
    let ixTarget = i === n - 1 ? 0 : letterToIndex[sent[i + 1]] // last step: end with END token

    lh = forwardIndex(G, model, ixSource, prev)
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

// let startTime
export function tick() {
  // if (!startTime) startTime = new Date().getTime()

  // sample sentence fromd data
  let sentix = randi(0, dataSents.length)
  let sent = dataSents[sentix]

  // evaluate cost function on a sentence
  let costStruct = costfun(model, sent)

  // use built up graph to compute backprop (set .dw fields in mats)
  costStruct.G.runBackprop()
  // perform param update
  solver.step(model, learningRate, regc, clipval)
  // let solverStats = solver.step(model, learningRate, regc, clipval)
  // $("#gradclip").text('grad clipped ratio: ' + solverStats.ratio_clipped)

  pplList.push(costStruct.ppl) // keep track of perplexity

  // evaluate now and then
  tickIter += 1

  // if (tickIter % 500 === 0) {
  // console.log('TIME ELAPSED PER 500 ITERATIONS: ', new Date().getTime() - startTime)
  // startTime = null
  // }

  if (tickIter % 50 === 0) {
    // draw samples
    // $('#samples').html('') // TODO: Show samples in the UI...for now just log them out
    for (let q = 0; q < 5; q++) {
      console.log(
        'NN output - sample:',
        predictSentence(model, true, sampleSoftmaxTemperature),
      )
      // var pred = predictSentence(model, true, sampleSoftmaxTemperature)
      // var pred_div = '<div class="apred">' + pred + '</div>'
      // $('#samples').append(pred_div)
    }
  }

  if (tickIter % 10 === 0) {
    // draw argmax prediction
    // TODO: Show this in the UI...for now just log it out
    console.log('NN output - argmax prediction:', predictSentence(model, false))
    // $('#argmax').html('')
    // var pred = predictSentence(model, false)
    // var pred_div = '<div class="apred">' + pred + '</div>'
    // $('#argmax').append(pred_div)

    // // keep track of perplexity
    // $('#epoch').text('epoch: ' + (tickIter / epochSize).toFixed(2))
    // $('#ppl').text('perplexity: ' + costStruct.ppl.toFixed(2))
    // $('#ticktime').text(
    //   'forw/bwd time per example: ' + tickTime.toFixed(1) + 'ms',
    // )

    // function median(values) {
    //   values.sort((a, b) => a - b) // OPT: Isn't this the default sort?
    //   const half = Math.floor(values.length / 2)
    //   return values.length % 2
    //     ? values[half]
    //     : (values[half - 1] + values[half]) / 2.0
    // }

    // TODO: Different solution for graph...maybe victory or something...or maybe antd has something
    // if (tickIter % 100 === 0) {
    // var median_ppl = median(pplList)
    // pplList = []
    // pplGraph.add(tickIter, median_ppl)
    // pplGraph.drawSelf(document.getElementById('pplgraph'))
    // }
  }
}

// This was commented out in his code...perhaps an unfinished idea?
// $('#gradcheck').click(gradCheck);
