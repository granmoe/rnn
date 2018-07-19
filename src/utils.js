import Mat from './Mat'

export const updateMats = func => (...mats) => {
  // TODO (someday): Assert that all mats have same length
  // I wonder if this whole loop and inner loop and everything could be one reduce?
  // prob would need vectorized ops like in numpy or R in order to decrease number of loops here
  for (let i = 0; i < mats[0].w.length; i++) {
    const weights = mats.reduce((weights, mat) => [...weights, mat.w[i], mat.dw[i]], [])
    const results = func(...weights)

    mats.forEach((mat, mIndex) => {
      const w = results[mIndex * 2]
      const dw = results[mIndex * 2 + 1]

      if (w !== undefined) mat.w[i] = w
      if (dw !== undefined) mat.dw[i] = dw
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

// sample argmax from w, assuming w are probabilities that sum to one
export function sampleIndex(w) {
  // TODO: Variable names here suck
  const r = randFloat(0, 1) // max value up to, but not including, 1
  let x = 0
  let i = 0

  while (x <= r) {
    x += w[i]
    i++
  }

  return i - 1
}

export function softmax(m) {
  const out = new Mat(m.rows, m.cols) // probability volume

  const [firstW, ...remainingW] = m.w
  let maxval = firstW
  remainingW.forEach(w => {
    if (w > maxval) maxval = w
  })

  let s = 0
  for (let i = 0; i < m.w.length; i++) {
    out.w[i] = Math.exp(m.w[i] - maxval)
    s += out.w[i]
  }

  out.updateW(w => w / s)

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
  // TODO: for...of is still just too slow even in the latest chrome :`-(
  // Maybe someday these can all change back to for (const i of range(len)) {}...
  for (let i = 0; i < len; i++) {
    yield arr.slice(i, i + windowSize)
  }
}
