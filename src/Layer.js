import { randFloat } from './utils'

// TODO: Change rows/cols to rowCount, colCount or numRows, numCols
export default class Layer {
  constructor(rows, cols) {
    this.rows = rows
    this.cols = cols
    this.weights = new Float64Array(rows * cols)
    this.gradients = new Float64Array(rows * cols)
    this.cachedGradients = new Float64Array(rows * cols)

    return this
  }

  indexToCoord(i) {
    return {
      row: Math.ceil((i + 1) / this.cols) - 1,
      col: ((i + 1) % this.cols || this.cols) - 1,
    }
  }

  updateWeights(func) {
    this.updateMat(func, this.weights)
    return this
  }

  updateGradients(func) {
    this.updateMat(func, this.gradients)
    return this
  }

  updateMat(func, mat) {
    mat.forEach((weight, i) => {
      // const { row, col } = this.indexToCoord(i)
      // mat[i] = func(weight, i, { row, col })
      mat[i] = func(weight, i, this.indexToCoord.bind(this))
    })
    return this
  }

  updateCache() {
    this.cachedGradients = new Float64Array(this.gradients)
  }

  clone() {
    // does not copy over values of gradients or cachedGradients
    const copy = new Layer(this.rows, this.cols)
    copy.weights = new Float64Array(this.weights)
    return copy
  }

  serialize() {
    // TODO IO
    return {}
  }
}

// TODO IO
export const matFromJSON = ({ rows, cols, weights }) => {
  const mat = new Layer(rows, cols)
  mat.weights = new Float64Array(Object.values(weights))
  return mat
}

// return Layer but filled with random numbers from gaussian
export const randLayer = (rows, cols, std = 0.08) =>
  new Layer(rows, cols).updateWeights(_ => randFloat(-std, std)) // kind of :P
