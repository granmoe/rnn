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
    out.updateW((w, i) => m.w[cols * ix + i])

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      if (input !== undefined) ix = input
      assert(ix >= 0 && ix < m.rows)

      out.updateW((w, i) => m.w[cols * ix + i])

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      for (let i = 0; i < cols; i++) {
        m.dw[cols * ix + i] += out.dw[i]
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
      m.updateDw((dw, i) => dw + (1 - out.w[i] * out.w[i]) * out.dw[i])
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
      m.updateDw((dw, i) => dw + out.w[i] * (1 - out.w[i]) * out.dw[i])
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
      m.updateDw((dw, i) => (dw + m.w[i] > 0 ? out.dw[i] : 0))
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
        dot += m1.w[n + row * m1.cols] * m2.w[n * m2.cols + col]
      }

      return dot
    })

    // out = dot product of m1 and m2
    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      out.updateW((_, i, mat) => {
        const { row, col } = mat.indexToCoord(i)

        let dot = 0
        for (let n = 0; n < m1.cols; n++) {
          dot += m1.w[n + row * m1.cols] * m2.w[n * m2.cols + col]
        }

        return dot
      })

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      out.dw.map((b, i) => {
        const { row, col } = out.indexToCoord(i)

        for (let n = 0; n < m1.cols; n++) {
          const m1i = n + row * m1.cols
          const m2i = n * m2.cols + col

          m1.dw[m1i] += m2.w[m2i] * b
          m2.dw[m2i] += m1.w[m1i] * b
        }
      })
      next()
    })

    // TODO NOW: The issue is that mul immediately returns a zero mat
    return out
  }

  add(m1, m2) {
    assert(m1.w.length === m2.w.length)
    const out = m1.clone()

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      // out.updateW((w, index) => m1.w[index] + m2.w[index]) todo: uncomment this
      out.updateW((w, index) => m1.w[index] + m2.w[index])

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      updateMats((m1w, m1dw, m2w, m2dw, outw, outdw) => {
        return [m1w, m1dw + outdw, m2w, m2dw + outdw]
      })(m1, m2, out)
      next()
    })

    return out
  }

  eltmul(m1, m2) {
    assert(m1.w.length === m2.w.length)
    let out = m1.clone()

    this.forwardFunctions.push(next => ({ input, doBackprop }) => {
      out.updateW((w, i) => w * m2.w[i])

      return next(out)
    })

    this.backwardFunctions.unshift(next => () => {
      updateMats((m1w, m1dw, m2w, m2dw, outw, outdw) => {
        return [m1w, m2w * outdw, m2w, m1w * outdw]
      })(m1, m2, out)
      next()
    })

    return out
  }
}
