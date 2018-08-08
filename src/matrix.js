import { assert, randFloat } from './utils'

// FIXME: Change rows/cols to rowCount, colCount or numRows, numCols
export default function createMat(rows, cols) {
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
      const copy = createMat(this.rows, this.cols)
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
  }
}

// return Layer but filled with random numbers from gaussian
export const createRandomMat = (rows, cols, std = 0.08) => {
  const randomLayer = createMat(rows, cols)
  updateWeights(randomLayer, _ => randFloat(-std, std)) // kind of :P
  return randomLayer
}

export const updateGradients = (mat, func) => {
  mat.gradients.forEach((gradient, i) => {
    mat.gradients[i] = func(gradient, i)
  })
}

export const updateWeights = (mat, func) => {
  mat.weights.forEach((weight, i) => {
    mat.weights[i] = func(weight, i)
  })
}

// TODO IO

// serialize() {}

// export const matFromJSON = ({ rows, cols, weights }) => {
//   const mat = createMat(rows, cols)
//   mat.weights = new Float64Array(Object.values(weights))
//   return mat
// }
