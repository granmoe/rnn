import { assert, updateMats } from './utils'
import createMat, {
  createRandomMat,
  cloneMat,
  matIndexToCoord,
  updateWeights,
  updateGradients,
} from './matrix'

// Does matrix ops, keeps track of backprop and performs backprop
export default class Graph {
  constructor() {
    this.backwardFunctions = []
    this.doBackprop = true
    this.layers = []
    this.nextLayerIndex = 0
  }

  getMat(opts) {
    if (opts.isLayer) return opts // existing layer passed in, don't use cache

    const { rows, cols, type = 'rand' } = opts

    let mat
    if (this.layers[this.nextLayerIndex]) {
      mat = this.layers[this.nextLayerIndex]
    } else {
      mat = type === 'rand' ? createRandomMat(rows, cols) : createMat(rows, cols)
      this.layers.push(mat)
    }

    this.nextLayerIndex++
    return mat
  }

  backward() {
    this.backwardFunctions.forEach(func => void func())
    this.backwardFunctions = []
  }

  rowPluck(mOpts, index) {
    const m = this.getMat(mOpts)
    // pluck a row of m with index index and return it as col vector
    const cols = m.cols
    const out = createMat(cols, 1)
    updateWeights(out, (_weight, i) => m.weights[cols * index + i])

    this.doBackprop &&
      this.backwardFunctions.unshift(() => {
        for (let i = 0; i < cols; i++) {
          m.gradients[cols * index + i] += out.gradients[i]
        }
      })

    return out
  }

  tanh(mOpts) {
    const m = this.getMat(mOpts)
    const out = cloneMat(m)
    updateWeights(out, Math.tanh) // tanh nonlinearity

    this.doBackprop &&
      this.backwardFunctions.unshift(() => {
        updateGradients(
          m,
          (gradient, i) =>
            gradient + (1 - out.weights[i] * out.weights[i]) * out.gradients[i],
        )
      })

    return out
  }

  sigmoid(mOpts) {
    const m = this.getMat(mOpts)
    const out = cloneMat(m)
    updateWeights(out, x => 1 / (1 + Math.exp(-x))) // sigmoid nonlinearity

    this.doBackprop &&
      this.backwardFunctions.unshift(() => {
        // grad for z = tanh(x) is (1 - z^2)
        updateGradients(
          m,
          (gradient, i) =>
            gradient + out.weights[i] * (1 - out.weights[i]) * out.gradients[i],
        )
      })

    return out
  }

  relu(mOpts) {
    const m = this.getMat(mOpts)
    const out = cloneMat(m)
    updateWeights(out, Math.max.bind(null, 0)) // sigmoid nonlinearity

    this.doBackprop &&
      this.backwardFunctions.unshift(() => {
        updateGradients(
          m,
          (gradient, i) => (gradient + m.weights[i] > 0 ? out.gradients[i] : 0),
        )
      })

    return out
  }

  mul(m1opts, m2opts) {
    const m1 = this.getMat(m1opts)
    const m2 = this.getMat(m2opts)
    assert(m1.cols === m2.rows, 'matmul dimensions misaligned')

    // out = dot product of m1 and m2
    const out = createMat(m1.rows, m2.cols)
    updateWeights(out, (_weight, i) => {
      const { row, col } = matIndexToCoord(out, i)
      let dot = 0
      for (let n = 0; n < m1.cols; n++) {
        dot += m1.weights[n + row * m1.cols] * m2.weights[n * m2.cols + col]
      }

      return dot
    })

    this.doBackprop &&
      this.backwardFunctions.unshift(() => {
        out.gradients.map((b, i) => {
          const { row, col } = matIndexToCoord(out, i)
          for (let n = 0; n < m1.cols; n++) {
            const m1i = n + row * m1.cols
            const m2i = n * m2.cols + col

            m1.gradients[m1i] += m2.weights[m2i] * b
            m2.gradients[m2i] += m1.weights[m1i] * b
          }
        })
      })

    return out
  }

  add(m1opts, m2opts) {
    const m1 = this.getMat(m1opts)
    const m2 = this.getMat(m2opts)
    assert(m1.weights.length === m2.weights.length)

    const out = cloneMat(m1)
    updateWeights(out, (_weight, index) => m1.weights[index] + m2.weights[index])

    this.doBackprop &&
      this.backwardFunctions.unshift(() => {
        updateMats((m1w, m1Gradient, m2w, m2Gradient, _outw, outGradient) => {
          return [m1w, m1Gradient + outGradient, m2w, m2Gradient + outGradient]
        })(m1, m2, out)
      })

    return out
  }

  eltmul(m1opts, m2opts) {
    const m1 = this.getMat(m1opts)
    const m2 = this.getMat(m2opts)
    assert(m1.weights.length === m2.weights.length)
    let out = cloneMat(m1)
    updateWeights(out, (weight, i) => weight * m2.weights[i])

    this.doBackprop &&
      this.backwardFunctions.unshift(() => {
        updateMats((m1w, _m1Gradient, m2w, _m2Gradient, _outw, outGradient) => {
          return [m1w, m2w * outGradient, m2w, m1w * outGradient]
        })(m1, m2, out)
      })

    return out
  }
}
