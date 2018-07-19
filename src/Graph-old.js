import { assert, updateMats } from './utils'
import Mat from './Mat'

// Does matrix ops, keeps track of backprop and performs backprop
export default class Graph {
  constructor({ doBackprop = true } = {}) {
    this.doBackprop = doBackprop
    this.backpropFuncs = [] // will be stored in reverse order of forward prop
  }

  runBackprop() {
    for (let i = 0; i < this.backpropFuncs.length; i++) {
      this.backpropFuncs[i]()
    }
  }

  rowPluck(m, ix) {
    assert(ix >= 0 && ix < m.rows)

    // pluck a row of m with index ix and return it as col vector
    const cols = m.cols
    const out = new Mat(cols, 1)
    out.updateW((w, i) => m.w[cols * ix + i])

    if (this.doBackprop) {
      this.backpropFuncs.unshift(() => {
        for (let i = 0; i < cols; i++) {
          m.dw[cols * ix + i] += out.dw[i]
        }
      })
    }

    return out
  }

  tanh(m) {
    const out = m.clone().updateW(Math.tanh) // tanh nonlinearity

    if (this.doBackprop) {
      this.backpropFuncs.unshift(() => {
        m.updateDw((dw, i) => dw + (1 - out.w[i] * out.w[i]) * out.dw[i])
      })
    }

    return out
  }

  sigmoid(m) {
    const out = m.clone().updateW(x => 1 / (1 + Math.exp(-x))) // sigmoid nonlinearity

    if (this.doBackprop) {
      this.backpropFuncs.unshift(() => {
        // grad for z = tanh(x) is (1 - z^2)
        m.updateDw((dw, i) => dw + out.w[i] * (1 - out.w[i]) * out.dw[i])
      })
    }

    return out
  }

  relu(m) {
    const out = m.clone().updateW(Math.max.bind(null, 0)) // sigmoid nonlinearity

    if (this.doBackprop) {
      this.backpropFuncs.unshift(() => {
        m.updateDw((dw, i) => (dw + m.w[i] > 0 ? out.dw[i] : 0))
      })
    }

    return out
  }

  mul(m1, m2) {
    assert(m1.cols === m2.rows, 'matmul dimensions misaligned')

    // out = dot product of m1 and m2
    const out = new Mat(m1.rows, m2.cols).updateW((_, i, mat) => {
      const { row, col } = mat.indexToCoord(i)

      let dot = 0
      for (let n = 0; n < m1.cols; n++) {
        dot += m1.w[n + row * m1.cols] * m2.w[n * m2.cols + col]
      }

      return dot
    })

    if (this.doBackprop) {
      this.backpropFuncs.unshift(() => {
        out.dw.map((b, i) => {
          const { row, col } = out.indexToCoord(i)

          for (let n = 0; n < m1.cols; n++) {
            const m1i = n + row * m1.cols
            const m2i = n * m2.cols + col

            m1.dw[m1i] += m2.w[m2i] * b
            m2.dw[m2i] += m1.w[m1i] * b
          }
        })
      })
    }

    return out
  }

  add(m1, m2) {
    assert(m1.w.length === m2.w.length)

    const out = m1.clone().updateW((w, index) => m1.w[index] + m2.w[index])

    if (this.doBackprop) {
      this.backpropFuncs.unshift(() => {
        updateMats((m1w, m1dw, m2w, m2dw, outw, outdw) => {
          return [m1w, m1dw + outdw, m2w, m2dw + outdw]
        })(m1, m2, out)
      })
    }

    return out
  }

  eltmul(m1, m2) {
    assert(m1.w.length === m2.w.length)

    let out = m1.clone().updateW((w, i) => w * m2.w[i])

    if (this.doBackprop) {
      this.backpropFuncs.unshift(() => {
        updateMats((m1w, m1dw, m2w, m2dw, outw, outdw) => {
          return [m1w, m2w * outdw, m2w, m1w * outdw]
        })(m1, m2, out)
      })
    }

    return out
  }
}