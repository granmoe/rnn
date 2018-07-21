import { assert, updateMats } from './utils'
import Mat from './Mat'

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
    this.output = null
  }

  forward(input) {
    if (!this.forwardFunc) {
      this.forwardFunc = composeMiddlewareFunctions(this.forwardFunctions)
    }

    return this.forwardFunc(input)
  }

  backward(input) {
    if (!this.backwardFunc) {
      this.backwardFunc = composeMiddlewareFunctions(this.backwardFunctions)
    }

    this.backwardFunc()
  }

  rowPluck(m, ix = 0) {
    // TODO: Make sure setting this to 0 doesn't cause any weird side-effects
    // need to have an arbitrary starting index to populate first output mat of this
    // pluck a row of m with index ix and return it as col vector
    const cols = m.cols
    const out = new Mat(cols, 1)
    out.updateW((weight, i) => m.weights[cols * ix + i])

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      if (input !== undefined) ix = input
      assert(ix >= 0 && ix < m.rows)

      out.updateW((weight, i) => m.weights[cols * ix + i])

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      for (let i = 0; i < cols; i++) {
        m.gradients[cols * ix + i] += out.gradients[i]
      }
      next()
    })

    return out
  }

  tanh(m) {
    const out = m.clone()

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      out.updateW(Math.tanh) // tanh nonlinearity

      return next(out)
    })

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
    const out = m.clone()

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      out.updateW(x => 1 / (1 + Math.exp(-x))) // sigmoid nonlinearity

      return next(out)
    })

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
    const out = m.clone()

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      out.updateW(Math.max.bind(null, 0)) // sigmoid nonlinearity

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      m.updateGradients(
        (gradient, i) => (gradient + m.weights[i] > 0 ? out.gradients[i] : 0),
      )
      next()
    })

    return out
  }

  mul(m1, m2) {
    // TODO: What if second is based on input result? Caller just has to pass it in
    assert(m1.cols === m2.rows, 'matmul dimensions misaligned')
    // Need to return something on first call
    // FIXME: How to make each graph op able to be used as first graph op in graph?
    // e.g. if not all args are there because there is no input yet
    // create placeholder input or something? Or sample first training example?
    const out = new Mat(m1.rows, m2.cols).updateW((_, i, mat) => {
      const { row, col } = mat.indexToCoord(i)

      let dot = 0
      for (let n = 0; n < m1.cols; n++) {
        dot += m1.weights[n + row * m1.cols] * m2.weights[n * m2.cols + col]
      }

      return dot
    })

    // out = dot product of m1 and m2
    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      out.updateW((_, i, mat) => {
        const { row, col } = mat.indexToCoord(i)

        let dot = 0
        for (let n = 0; n < m1.cols; n++) {
          dot += m1.weights[n + row * m1.cols] * m2.weights[n * m2.cols + col]
        }

        return dot
      })

      return next(out)
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

    // TODO NOW: The issue is that mul immediately returns a zero mat
    return out
  }

  add(m1, m2) {
    assert(m1.weights.length === m2.weights.length)
    const out = m1.clone()

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      // out.updateW((weight, index) => m1.weights[index] + m2.weights[index]) todo: uncomment this
      out.updateW((weight, index) => m1.weights[index] + m2.weights[index])

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      updateMats((m1w, m1Gradient, m2w, m2Gradient, outw, outGradient) => {
        return [m1w, m1Gradient + outGradient, m2w, m2Gradient + outGradient]
      })(m1, m2, out)
      next()
    })

    return out
  }

  eltmul(m1, m2) {
    assert(m1.weights.length === m2.weights.length)
    let out = m1.clone()

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      out.updateW((weight, i) => weight * m2.weights[i])

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      updateMats((m1w, m1Gradient, m2w, m2Gradient, outw, outGradient) => {
        return [m1w, m2w * outGradient, m2w, m1w * outGradient]
      })(m1, m2, out)
      next()
    })

    return out
  }
}
