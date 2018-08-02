import { randFloat } from './utils'

// FIXME: Change rows/cols to rowCount, colCount or numRows, numCols
export default function createLayer(rows, cols) {
  const length = rows * cols

  const layer = {
    isLayer: true,
    rows,
    cols,
    length: rows * cols,
    weights: new Float64Array(length),
    gradients: new Float64Array(length),
    cachedGradients: new Float64Array(length),
    indexToCoord(i) {
      return {
        row: Math.ceil((i + 1) / cols) - 1,
        col: ((i + 1) % cols || cols) - 1,
      }
    },
    updateWeights(func) {
      layer.updateMat(func, layer.weights)
      return layer
    },
    updateGradients(func) {
      layer.updateMat(func, layer.gradients)
    },
    updateMat(func, mat) {
      mat.forEach((weight, i) => {
        mat[i] = func(weight, i, layer.indexToCoord)
      })
    },
    resetGradients() {
      layer.gradients = new Float64Array(length)
    },
    clone() {
      // does not copy over values of gradients or cachedGradients
      const copy = createLayer(rows, cols)
      copy.weights = new Float64Array(layer.weights)
      return copy
    },
    serialize() {}, // TODO IO
  }

  return layer
}

// TODO IO
export const matFromJSON = ({ rows, cols, weights }) => {
  const mat = createLayer(rows, cols)
  mat.weights = new Float64Array(Object.values(weights))
  return mat
}

// return Layer but filled with random numbers from gaussian
export const randLayer = (rows, cols, std = 0.08) =>
  createLayer(rows, cols).updateWeights(_ => randFloat(-std, std)) // kind of :P
