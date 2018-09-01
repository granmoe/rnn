import GPU from 'gpu.js'
import { assert, updateMats } from './utils'
import createMat, { createRandomMat } from './matrix'

const gpu = new GPU()

// Does matrix ops, keeps track of backprop and performs backprop
export default function createGraph() {
  let backwardFunctions = []

  return {
    layers: [],
    doBackprop: true,
    nextLayerIndex: 0,

    getMat(opts) {
      if (opts.isLayer) return opts // existing layer passed in, don't use cache

      const { rows, cols, type = 'rand' } = opts

      let mat
      if (this.layers[this.nextLayerIndex]) {
        mat = this.layers[this.nextLayerIndex]
      } else {
        mat =
          type === 'rand' ? createRandomMat({ rows, cols }) : createMat({ rows, cols })
        this.layers.push(mat)
      }

      this.nextLayerIndex++
      return mat
    },
    backward() {
      backwardFunctions.forEach(func => void func())
      backwardFunctions = []
    },
    rowPluck(mOpts, index) {
      const m = this.getMat(mOpts)
      // pluck a row of m with index index and return it as col vector
      const cols = m.cols
      const out = createMat({ rows: cols, cols: 1 }).updateWeights(
        (_weight, i) => m.weights[cols * index + i],
      )

      this.doBackprop &&
        backwardFunctions.unshift(() => {
          for (let i = 0; i < cols; i++) {
            m.gradients[cols * index + i] += out.gradients[i]
          }
        })

      return out
    },
    tanh(mOpts) {
      const m = this.getMat(mOpts)
      const out = m.clone().updateWeights(Math.tanh) // tanh nonlinearity

      this.doBackprop &&
        backwardFunctions.unshift(() => {
          m.updateGradients(
            (gradient, i) =>
              gradient + (1 - out.weights[i] * out.weights[i]) * out.gradients[i],
          )
        })

      return out
    },
    sigmoid(mOpts) {
      const m = this.getMat(mOpts)
      const out = m.clone().updateWeights(x => 1 / (1 + Math.exp(-x))) // sigmoid nonlinearity

      this.doBackprop &&
        backwardFunctions.unshift(() => {
          // grad for z = tanh(x) is (1 - z^2)
          m.updateGradients(
            (gradient, i) =>
              gradient + out.weights[i] * (1 - out.weights[i]) * out.gradients[i],
          )
        })

      return out
    },
    relu(mOpts) {
      const m = this.getMat(mOpts)
      const out = m.clone().updateWeights(Math.max.bind(null, 0)) // sigmoid nonlinearity

      this.doBackprop &&
        backwardFunctions.unshift(() => {
          m.updateGradients(
            (gradient, i) => (gradient + m.weights[i] > 0 ? out.gradients[i] : 0),
          )
        })

      return out
    },
    mul(m1opts, m2opts) {
      const m1 = this.getMat(m1opts)
      const m2 = this.getMat(m2opts)
      assert(m1.cols === m2.rows, 'matmul dimensions misaligned')

      const out = createMat({ rows: m1.rows, cols: m2.cols })

      const gpuMul = gpu.createKernel(
        function(m1w, m1cols, m2w, m2cols, outCols) {
          const row = Math.floor(this.thread.x / outCols)
          const col = this.thread.x - row * this.thread.x

          let dot = 0
          for (let n = 0; n < this.constants.size; n++) {
            dot += m1w[n + row * m1cols] * m2w[n * m2cols + col]
          }

          return dot
        },
        { output: [out.length], constants: { size: m1.cols } },
      )

      out.weights = new Float32Array(
        gpuMul(m1.weights, m1.cols, m2.weights, m2.cols, out.cols),
      )

      // out = dot product of m1 and m2
      // out.updateWeights((_weight, i, indexToCoord) => {
      //   const { row, col } = indexToCoord(i)
      //   let dot = 0
      //   for (let n = 0; n < m1.cols; n++) {
      //     dot += m1.weights[n + row * m1.cols] * m2.weights[n * m2.cols + col]
      //   }

      //   return dot
      // })

      this.doBackprop &&
        backwardFunctions.unshift(() => {
          out.gradients.map((b, i) => {
            const { row, col } = out.indexToCoord(i)
            for (let n = 0; n < m1.cols; n++) {
              const m1i = n + row * m1.cols
              const m2i = n * m2.cols + col

              m1.gradients[m1i] += m2.weights[m2i] * b
              m2.gradients[m2i] += m1.weights[m1i] * b
            }
          })
        })

      return out
    },
    add(m1opts, m2opts) {
      const m1 = this.getMat(m1opts)
      const m2 = this.getMat(m2opts)
      assert(m1.weights.length === m2.weights.length)

      const out = m1
        .clone()
        .updateWeights((_weight, index) => m1.weights[index] + m2.weights[index])

      this.doBackprop &&
        backwardFunctions.unshift(() => {
          updateMats((m1w, m1Gradient, m2w, m2Gradient, _outw, outGradient) => {
            return [m1w, m1Gradient + outGradient, m2w, m2Gradient + outGradient]
          })(m1, m2, out)
        })

      return out
    },
    eltmul(m1opts, m2opts) {
      const m1 = this.getMat(m1opts)
      const m2 = this.getMat(m2opts)
      assert(m1.weights.length === m2.weights.length)
      let out = m1.clone().updateWeights((weight, i) => weight * m2.weights[i])

      this.doBackprop &&
        backwardFunctions.unshift(() => {
          updateMats((m1w, _m1Gradient, m2w, _m2Gradient, _outw, outGradient) => {
            return [m1w, m2w * outGradient, m2w, m1w * outGradient]
          })(m1, m2, out)
        })

      return out
    },
  }
}
