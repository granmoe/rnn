import { assert, updateMats } from './utils'
import Layer from './Layer'

const identityFunc = a => a

export const composeMiddlewareFunctions = (funcs, onFinishedCallback = identityFunc) =>
  funcs.reduceRight(
    (currentFunc, precedingFunc) => precedingFunc(currentFunc),
    onFinishedCallback,
  )

// Does matrix ops, keeps track of backprop and performs backprop
export default class Graph {
  constructor() {
    this.forwardFunctions = []
    this.backwardFunctions = []
  }

  backward() {
    if (!this.backwardFunc) {
      this.backwardFunc = composeMiddlewareFunctions(this.backwardFunctions)
    }

    this.backwardFunc()
  }

  rowPluck(m, index) {
    // pluck a row of m with index index and return it as col vector
    const cols = m.cols
    const out = new Layer(cols, 1)
    out.updateWeights((_weight, i) => m.weights[cols * index + i])

    this.backwardFunctions.unshift(next => () => {
      for (let i = 0; i < cols; i++) {
        m.gradients[cols * index + i] += out.gradients[i]
      }
      next()
    })

    return out
  }

  tanh(m) {
    const out = m.clone().updateWeights(Math.tanh) // tanh nonlinearity

    this.backwardFunctions.unshift(next => () => {
      m.updateGradients(
        (gradient, i) =>
          gradient + (1 - out.weights[i] * out.weights[i]) * out.gradients[i],
      )
      next()
    })

    return out
  }

  sigmoid(m) {
    const out = m.clone().updateWeights(x => 1 / (1 + Math.exp(-x))) // sigmoid nonlinearity

    this.backwardFunctions.unshift(next => () => {
      // grad for z = tanh(x) is (1 - z^2)
      m.updateGradients(
        (gradient, i) =>
          gradient + out.weights[i] * (1 - out.weights[i]) * out.gradients[i],
      )
      next()
    })

    return out
  }

  relu(m) {
    const out = m.clone().updateWeights(Math.max.bind(null, 0)) // sigmoid nonlinearity

    this.backwardFunctions.unshift(next => () => {
      m.updateGradients(
        (gradient, i) => (gradient + m.weights[i] > 0 ? out.gradients[i] : 0),
      )
      next()
    })

    return out
  }

  mul(m1, m2) {
    assert(m1.cols === m2.rows, 'matmul dimensions misaligned')

    // out = dot product of m1 and m2
    const out = new Layer(m1.rows, m2.cols).updateWeights((_weight, i, indexToCoord) => {
      const { row, col } = indexToCoord(i)
      let dot = 0
      for (let n = 0; n < m1.cols; n++) {
        dot += m1.weights[n + row * m1.cols] * m2.weights[n * m2.cols + col]
      }

      return dot
    })

    this.backwardFunctions.unshift(next => () => {
      out.gradients.map((b, i) => {
        const { row, col } = out.indexToCoord(i)

        for (let n = 0; n < m1.cols; n++) {
          const m1i = n + row * m1.cols
          const m2i = n * m2.cols + col

          m1.gradients[m1i] += m2.weights[m2i] * b
          m2.gradients[m2i] += m1.weights[m1i] * b
        }
      })
      next()
    })

    return out
  }

  add(m1, m2) {
    assert(m1.weights.length === m2.weights.length)
    const out = m1
      .clone()
      .updateWeights((_weight, index) => m1.weights[index] + m2.weights[index])

    this.backwardFunctions.unshift(next => () => {
      updateMats((m1w, m1Gradient, m2w, m2Gradient, _outw, outGradient) => {
        return [m1w, m1Gradient + outGradient, m2w, m2Gradient + outGradient]
      })(m1, m2, out)
      next()
    })

    return out
  }

  eltmul(m1, m2) {
    assert(m1.weights.length === m2.weights.length)
    let out = m1.clone().updateWeights((weight, i) => weight * m2.weights[i])

    this.backwardFunctions.unshift(next => () => {
      updateMats((m1w, _m1Gradient, m2w, _m2Gradient, _outw, outGradient) => {
        return [m1w, m2w * outGradient, m2w, m1w * outGradient]
      })(m1, m2, out)
      next()
    })

    return out
  }
}
