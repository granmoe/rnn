import Mat from './Mat'

export const updateMats = func => (...mats) => {
  // TODO: Assert that all mats have same length
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
export function maxi(w) {
  // argmax of array w
  let maxv = w[0]
  let maxix = 0
  w.forEach((w, i) => {
    if (w > maxv) {
      maxv = w
      maxix = i
    }
  })

  return maxix
}

export function samplei(w) {
  // sample argmax from w, assuming w are probabilities that sum to one
  let r = randf(0, 1) // max value up to, but not including, 1
  let x = 0
  let i = 0

  while (x <= r) {
    x += w[i]
    i++
  }

  return i
}

export function softmax(m) {
  let out = new Mat(m.n, m.d) // probability volume

  let maxval = -999999
  m.w.forEach(w => {
    if (w > maxval) maxval = w
  })

  let s = 0.0
  for (let i = 0, n = m.w.length; i < n; i++) {
    out.w[i] = Math.exp(m.w[i] - maxval)
    s += out.w[i]
  }

  out.updateW(w => w / s)

  // no backward pass here needed since we will use the computed
  // probabilities outside to set gradients directly on m
  return out
}

