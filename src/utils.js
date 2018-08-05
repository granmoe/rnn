import createLayer, { updateWeights } from './Layer'

export const updateMats = func => (...mats) => {
  // FIXME: Assert that all mats have same length
  // I wonder if this whole loop and inner loop and everything could be one reduce?
  // prob would need vectorized ops like in numpy or R in order to decrease number of loops here
  for (let i = 0; i < mats[0].weights.length; i++) {
    const weights = mats.reduce(
      (allWeights, mat) => [...allWeights, mat.weights[i], mat.gradients[i]],
      [],
    )
    const results = func(...weights)

    mats.forEach((mat, mIndex) => {
      const weight = results[mIndex * 2]
      const gradient = results[mIndex * 2 + 1]

      if (weight !== undefined) mat.weights[i] = weight
      if (gradient !== undefined) mat.gradients[i] = gradient
    })
  }
}

export function assert(condition, message = 'Assertion failed') {
  if (!condition) {
    throw new Error(message)
  }
}

export const randFloat = (a, b) => Math.random() * (b - a) + a

export const randInt = (a, b) => Math.floor(randFloat(a, b))

// I think the original code was only iterating through the second-to-last item
// in order to make predictSentence work (to account for end token)
// Maybe this func should be simplified to just return max ix of array
// And caller should be responsible for passing in the array that
// Already excludes the last item
export const maxIndex = weights =>
  weights
    .slice(0, weights.length - 1)
    .reduce(
      (maxIndex, weight, weightIndex, weights) =>
        weight > weights[maxIndex] ? weightIndex : maxIndex,
      0,
    )

// sample argmax from weight, assuming weight are probabilities that sum to one
export function sampleIndex(weights) {
  // FIXME: Variable names here suck
  const r = randFloat(0, 1) // max value up to, but not including, 1
  let x = 0
  let i = 0

  while (x <= r) {
    x += weights[i]
    i++
  }

  return i - 1
}

export function softmax(m) {
  const out = createLayer(m.rows, m.cols) // probability volume

  const [firstW, ...remainingW] = m.weights
  let maxval = firstW
  remainingW.forEach(weight => {
    if (weight > maxval) maxval = weight
  })

  let s = 0
  for (let i = 0; i < m.weights.length; i++) {
    out.weights[i] = Math.exp(m.weights[i] - maxval)
    s += out.weights[i]
  }

  updateWeights(out, weight => weight / s)

  // no backward pass here needed since we will use the computed
  // probabilities outside to set gradients directly on m
  return out
}

export function* range(maxOrStart, max) {
  const range = Array.from(
    { length: max === undefined ? maxOrStart : max },
    (_, key) => (max === undefined ? key : key + maxOrStart),
  )

  for (const num of range) {
    yield num
  }
}

export function* slidingWindow(windowSize, arr) {
  const len = arr.length - (windowSize - 1)
  // FIXME: for...of is still just too slow even in the latest chrome :`-(
  // Maybe someday these can all change back to for (const i of range(len)) {}...
  for (let i = 0; i < len; i++) {
    yield arr.slice(i, i + windowSize)
  }
}
