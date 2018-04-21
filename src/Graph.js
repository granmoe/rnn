import { assert, updateMats } from './utils'
import Mat from './Mat'

// Transformer definitions
export default class Graph {
  constructor(needsBackprop = true) {
    this.needsBackprop = needsBackprop

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards through this array and invoke each one
    this.backprop = []
  }

  backward() {
    for (let i = this.backprop.length - 1; i >= 0; i--) {
      this.backprop[i]() // tick! <- What does this mean?
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
      this.backprop.push(backward)
    }
    return out
  }

  tanh(m) {
    const out = m.clone().updateW(Math.tanh) // tanh nonlinearity

    if (this.needsBackprop) {
      this.backprop.push(() => {
        m.updateDw((dw, i) => dw + (1 - out.w[i] * out.w[i]) * out.dw[i])
      })
    }

    return out
  }

  sigmoid(m) {
    const out = m.clone().updateW(sig) // sigmoid nonlinearity

    if (this.needsBackprop) {
      this.backprop.push(() => {
        // grad for z = tanh(x) is (1 - z^2)
        m.updateDw((dw, i) => dw + out.w[i] * (1 - out.w[i]) * out.dw[i])
      })
    }

    return out
  }

  relu(m) {
    const out = m.clone().updateW(Math.max.bind(null, 0)) // sigmoid nonlinearity

    if (this.needsBackprop) {
      this.backprop.push(() => {
        m.updateDw((dw, i) => (dw + m.w[i] > 0 ? out.dw[i] : 0))
      })
    }

    return out
  }

  // TODO NEXT: refactor this
  mul(m1, m2) {
    function backward() {
      for (let i = 0; i < m1.rows; i++) {
        // loop over rows of m1
        for (let j = 0; j < m2.cols; j++) {
          // loop over cols of m2
          for (let k = 0; k < m1.cols; k++) {
            // dot product loop
            let b = out.dw[cols * i + j]
            m1.dw[m1.cols * i + k] += m2.w[m2.cols * k + j] * b
            m2.dw[m2.cols * k + j] += m1.w[m1.cols * i + k] * b
          }
        }
      }
    }

    // multiply matrices m1 * m2
    assert(m1.cols === m2.rows, 'matmul dimensions misaligned')

    let rows = m1.rows
    let cols = m2.cols
    let out = new Mat(rows, cols)
    for (let i = 0; i < m1.rows; i++) {
      // loop over rows of m1
      for (let j = 0; j < m2.cols; j++) {
        // loop over cols of m2
        let dot = 0.0
        for (let k = 0; k < m1.cols; k++) {
          // dot product loop
          dot += m1.w[m1.cols * i + k] * m2.w[m2.cols * k + j]
        }
        out.w[cols * i + j] = dot
      }
    }

    if (this.needsBackprop) {
      this.backprop.push(backward)
    }
    return out
  }

  add(m1, m2) {
    assert(m1.w.length === m2.w.length)

    const out = m1.clone().updateW((w, index) => m1.w[index] + m2.w[index])

    if (this.needsBackprop) {
      this.backprop.push(() => {
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
      this.backprop.push(() => {
        updateMats((m1w, m1dw, m2w, m2dw, outw, outdw) => {
          return [m1w, m2w * outdw, m2w, m1w * outdw]
        })(m1, m2, out)
      })
    }

    return out
  }
}

const sig = x => 1.0 / (1 + Math.exp(-x))