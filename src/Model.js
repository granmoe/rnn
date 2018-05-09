import Solver from './Solver'
import Graph from './Graph'
import { randi, softmax, maxi, samplei } from './utils'
import { initRNN, initLSTM, forwardRNN, forwardLSTM } from './RNN'

// optimization
const regc = 0.000001 // L2 regularization strength
const learningRate = 0.01 // learning rate
const clipval = 5.0 // clip gradients at this value
// prediction params
let sampleSoftmaxTemperature = 1.0 // how peaky model predictions should be
let maxCharsGen = 100 // max length of generated sentences

// various global var inits
let solver = new Solver() // should be class because it needs memory for step caches
// let pplGraph = new Rvis()

let lh, logprobs, probs

function initVocab(sentences, charCountThreshold = 1) {
  // go over all characters and keep track of all unique ones seen
  const charCounts = [...sentences.join('')].reduce((counts, char) => {
    counts[char] = counts[char] ? counts[char] + 1 : (counts[char] = 1)
    return counts
  }, {})

  // NOTE: start at nextIndex at 1 because we will have START and END tokens!
  // that is, START token will be index 0 in model letter vectors
  // and END token will be index 0 in the next character softmax
  const initialVocabData = {
    letterToIndex: {},
    indexToLetter: {},
    vocab: [],
    nextIndex: 1,
    inputSize: 1,
  }

  return Object.entries(charCounts).reduce((result, [char, count]) => {
    if (count >= charCountThreshold) {
      result.vocab.push(char)
      result.letterToIndex[char] = result.nextIndex
      result.indexToLetter[result.nextIndex] = char
      result.nextIndex += 1
      result.inputSize += 1
    }

    return result
  }, initialVocabData)
  // epochSize = sentences.length
  // TODO: Show this in the UI
  // $('#prepro_status').text(
  // 'found ' + vocab.length + ' distinct characters: ' + vocab.join(''),
  // )
}

// TODO: This is all a mess, obviously. It's just in an awkward stage of refactoring.
export function createModel(hyperParams) {
  const { type, hiddenSizes, letterSize, input } = hyperParams
  const sentences = input.split('\n').map(str => str.trim())
  const { letterToIndex, indexToLetter, vocab, inputSize } = initVocab(
    sentences,
    hyperParams.charCountThreshold,
  )

  if (type === 'rnn') {
    let rnn = initRNN(letterSize, hiddenSizes, inputSize)
    return {
      ...rnn,
      hyperParams,
      sentences,
      letterToIndex,
      indexToLetter,
      vocab,
    }
  } else {
    let lstm = initLSTM(letterSize, hiddenSizes, inputSize)
    return {
      ...lstm,
      hyperParams,
      sentences,
      letterToIndex,
      indexToLetter,
      vocab,
    }
  }
}

function forwardIndex(G, model, ix, prev) {
  const x = G.rowPluck(model['Wil'], ix)
  // forward prop the sequence learner
  return model.hyperParams.type === 'rnn'
    ? forwardRNN(G, model, x, prev)
    : forwardLSTM(G, model, x, prev)
}

function predictSentence(model, sample = false, temperature = 1.0) {
  let G = new Graph(false)
  let s = ''
  let prev = {}
  while (true) {
    // RNN tick
    let ix = s.length === 0 ? 0 : model.letterToIndex[s[s.length - 1]]
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

    let letter = model.indexToLetter[ix]
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
    let ixSource = i === -1 ? 0 : model.letterToIndex[sent[i]] // first step: start with START token
    let ixTarget = i === n - 1 ? 0 : model.letterToIndex[sent[i + 1]] // last step: end with END token

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

let tickIter = 0
export function train(model) {
  // sample sentence from data
  const sentence = model.sentences[randi(0, model.sentences.length)]
  // evaluate cost function on a sentence
  let costStruct = costfun(model, sentence)

  // use built up graph to compute backprop (set .dw fields in mats)
  costStruct.G.runBackprop()
  // perform param update
  solver.step(model, learningRate, regc, clipval)
  // let solverStats = solver.step(model, learningRate, regc, clipval)
  // $("#gradclip").text('grad clipped ratio: ' + solverStats.ratio_clipped)

  // pplList.push(costStruct.ppl) // keep track of perplexity

  tickIter += 1

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
