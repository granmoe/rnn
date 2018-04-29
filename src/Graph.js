import { assert, updateMats } from './utils'
import Mat from './Mat'

// Does matrix ops, keeps track of backprop and performs backprop

// Transformer definitions
export default class Graph {
  constructor(needsBackprop = true) {
    this.needsBackprop = needsBackprop

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards through this array and invoke each one
    // rename to backpropFuncs
    this.backpropFuncs = []
  }

  runBackprop() {
    for (let i = this.backpropFuncs.length - 1; i >= 0; i--) {
      this.backpropFuncs[i]() // tick! <- What does this mean?
    }
  }

  rowPluck(m, ix) {
    function backward() {
      for (let i = 0; i < cols; i++) {
        m.dw[cols * ix + i] += out.dw[i]
      }
    }

    assert(ix >= 0 && ix < m.rows)

    // pluck a row of m with index ix and return it as col vector
    let cols = m.cols
    let out = new Mat(cols, 1)
    out.updateW((w, i) => m.w[cols * ix + i])

    if (this.needsBackprop) {
      this.backpropFuncs.push(backward)
    }
    return out
  }

  tanh(m) {
    const out = m.clone().updateW(Math.tanh) // tanh nonlinearity

    if (this.needsBackprop) {
      this.backpropFuncs.push(() => {
        m.updateDw((dw, i) => dw + (1 - out.w[i] * out.w[i]) * out.dw[i])
      })
    }

    return out
  }

  sigmoid(m) {
    const out = m.clone().updateW(sig) // sigmoid nonlinearity

    if (this.needsBackprop) {
      this.backpropFuncs.push(() => {
        // grad for z = tanh(x) is (1 - z^2)
        m.updateDw((dw, i) => dw + out.w[i] * (1 - out.w[i]) * out.dw[i])
      })
    }

    return out
  }

  relu(m) {
    const out = m.clone().updateW(Math.max.bind(null, 0)) // sigmoid nonlinearity

    if (this.needsBackprop) {
      this.backpropFuncs.push(() => {
        m.updateDw((dw, i) => (dw + m.w[i] > 0 ? out.dw[i] : 0))
      })
    }

    return out
  }

  // TODO NEXT: refactor this using updateMats
  mul(m1, m2) {
    function backward() {
      for (let i = 0; i < m1.rows; i++) {
        // loop over rows of m1
        for (let j = 0; j < m2.cols; j++) {
          // loop over cols of m2
          for (let k = 0; k < m1.cols; k++) {
            // dot product loop
            let b = out.dw[m2.cols * i + j]
            const m2i = m2.cols * k + j
            const m1i = m1.cols * i + k
            // const { row: m1r, col: m1col } = m1.indexToCoord(m1i)
            // const { row: m2r, col: m2col } = m1.indexToCoord(m2i)
            // const x = m1r + m1col + m2r + m2col

            m1.dw[m1i] += m2.w[m2i] * b
            m2.dw[m2i] += m1.w[m1i] * b
          }
        }
      }
    }

    assert(m1.cols === m2.rows, 'matmul dimensions misaligned')

    // multiply matrices m1 * m2
    let out = new Mat(m1.rows, m2.cols).updateW((_, i, mat) => {
      const { row, col } = mat.indexToCoord(i)

      let dot = 0
      for (let n = 0; n < m1.cols; n++) {
        dot += m1.w[n + row * m1.cols] * m2.w[n * m2.cols + col]
      }

      return dot
    })

    if (this.needsBackprop) {
      this.backpropFuncs.push(backward)
    }

    return out
  }

  add(m1, m2) {
    assert(m1.w.length === m2.w.length)

    const out = m1.clone().updateW((w, index) => m1.w[index] + m2.w[index])

    if (this.needsBackprop) {
      this.backpropFuncs.push(() => {
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

    if (this.needsBackprop) {
      this.backpropFuncs.push(() => {
        updateMats((m1w, m1dw, m2w, m2dw, outw, outdw) => {
          return [m1w, m2w * outdw, m2w, m1w * outdw]
        })(m1, m2, out)
      })
    }

    return out
  }
}

const sig = x => 1.0 / (1 + Math.exp(-x))
