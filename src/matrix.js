import { assert, randFloat } from './utils'

// FIXME: Change rows/cols to rowCount, colCount or numRows, numCols
export default function createMat({ rows, cols }) {
  const length = rows * cols

  return {
    isLayer: true,
    rows,
    cols,
    length: rows * cols,
    weights: new Float64Array(length),
    gradients: new Float64Array(length),
    cachedGradients: new Float64Array(length),
    clone() {
      const copy = createMat({ rows: this.rows, cols: this.cols })
      copy.weights = new Float64Array(this.weights)
      return copy
    },
    resetGradients() {
      this.gradients = new Float64Array(this.length)
    },
    indexToCoord(i) {
      assert(i < this.length, 'index greater than matrix length')
      return {
        row: Math.ceil((i + 1) / this.cols) - 1,
        col: ((i + 1) % this.cols || this.cols) - 1,
      }
    },
    updateGradients(func) {
      this.gradients.forEach((gradient, i) => {
        this.gradients[i] = func(gradient, i)
      })
    },
    updateWeights(func) {
      this.weights.forEach((weight, i) => {
        this.weights[i] = func(weight, i, this.indexToCoord.bind(this))
      })
      return this
    },
  }
}

// return Layer but filled with random numbers from gaussian
export const createRandomMat = ({ rows, cols }, std = 0.08) =>
  createMat({ rows, cols }).updateWeights(_ => randFloat(-std, std)) // kind of :P

// TODO IO
// serialize() {}
// export const matFromJSON = ({ rows, cols, weights }) => {
//   const mat = createMat({ rows, cols })
//   mat.weights = new Float64Array(Object.values(weights))
//   return mat
// }
