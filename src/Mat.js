import { randFloat } from './utils'

// TODO: Consider making weights and gradients each a Mat
// Then make a composite data structure from these (could be called "layer")

// TODO: Change rows/cols to rowCount, colCount or numRows, numCols
export default class Mat {
  constructor(rows, cols) {
    this.rows = rows
    this.cols = cols
    this.weights = zeros(rows * cols)
    this.gradients = zeros(rows * cols)

    return this
  }

  indexToCoord(i) {
    return {
      row: Math.ceil((i + 1) / this.cols) - 1,
      col: ((i + 1) % this.cols || this.cols) - 1,
    }
  }

  updateW(func) {
    this.weights.forEach((weight, i) => {
      this.weights[i] = func(weight, i, this)
    })
    return this
  }

  updateGradients(func) {
    this.gradients = this.gradients.map(func)
    return this
  }

  clone() {
    // does not copy over gradients
    const copy = new Mat(this.rows, this.cols)
    copy.weights = new Float64Array(this.weights)
    return copy
  }

  serialize() {
    return {
      rows: this.rows,
      cols: this.cols,
      weights: this.weights,
    }
  }
}

export const matFromJSON = ({ rows, cols, weights }) => {
  const mat = new Mat(rows, cols)
  mat.weights = new Float64Array(Object.values(weights))
  return mat
}

// return Mat but filled with random numbers from gaussian
export const randMat = (rows, cols, std = 0.08) =>
  new Mat(rows, cols).updateW(_ => randFloat(-std, std)) // kind of :P

const zeros = count => new Float64Array(count)
