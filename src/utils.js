import Mat from './Mat'

export const updateMats = func => (...mats) => {
  // TODO (someday): Assert that all mats have same length
  // I wonder if this whole loop and inner loop and everything could be one reduce?
  // prob would need vectorized ops like in numpy or R in order to decrease number of loops here
  for (let i = 0; i < mats[0].w.length; i++) {
    const weights = mats.reduce(
      (weights, mat) => [...weights, mat.w[i], mat.dw[i]],
      [],
    )

    const results = func(...weights)

    mats.forEach((mat, mIndex) => {
      const w = results[mIndex * 2]
      const dw = results[mIndex * 2 + 1]

      if (w !== undefined) {
        mat.w[i] = results[mIndex * 2]
      }
      if (dw !== undefined) {
        mat.dw[i] = results[mIndex * 2 + 1]
      }
    })
  }
}

export function assert(condition, message = 'Assertion failed') {
  if (!condition) {
    throw new Error(message)
  }
}

export const randf = (a, b) => Math.random() * (b - a) + a

export const randi = (a, b) => Math.floor(Math.random() * (b - a) + a)

// TODO: Variable names here suck
// I think the original code was only iterating through the second-to-last item
// in order to make predictSentence work (to account for end token)
// Maybe this func should be simplified to just return max ix of array
// And caller should be responsible for passing in the array that
// Already excludes the last item
export const maxi = weights =>
  weights
    .slice(0, weights.length - 1)
    .reduce(
      (maxIndex, weight, weightIndex, weights) =>
        weight > weights[maxIndex] ? weightIndex : maxIndex,
      0,
    )

export function samplei(w) {
  // sample argmax from w, assuming w are probabilities that sum to one
  let r = randf(0, 1) // max value up to, but not including, 1
  let x = 0
  let i = 0

  while (x <= r) {
    x += w[i]
    i++
  }

  return i - 1
}

export function softmax(m) {
  let out = new Mat(m.rows, m.cols) // probability volume

  let maxval = -999999
  m.w.forEach(w => {
    if (w > maxval) maxval = w
  })

  let s = 0.0
  for (let i = 0; i < m.w.length; i++) {
    out.w[i] = Math.exp(m.w[i] - maxval)
    s += out.w[i]
  }

  out.updateW(w => w / s)

  // no backward pass here needed since we will use the computed
  // probabilities outside to set gradients directly on m
  return out
}

export const repeat = (count, func) => {
  for (let i = 1; i <= count; i++) {
    if (i === count) {
      return func(i) // return the result of the last call
    } else {
      func(i)
    }
  }
}
