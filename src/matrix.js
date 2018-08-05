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
  }
}

// return Layer but filled with random numbers from gaussian
export const createRandomMat = (rows, cols, std = 0.08) => {
  const randomLayer = createMat(rows, cols)
  updateWeights(randomLayer, _ => randFloat(-std, std)) // kind of :P
  return randomLayer
}

export const cloneMat = mat => {
  // does not copy over values of gradients or cachedGradients
  const copy = createMat(mat.rows, mat.cols)
  copy.weights = new Float64Array(mat.weights)
  return copy
}

export const resetGradients = mat => {
  mat.gradients = new Float64Array(mat.length)
}

export const matIndexToCoord = (mat, i) => {
  assert(i < mat.length, 'index greater than matrix length')
  return {
    row: Math.ceil((i + 1) / mat.cols) - 1,
    col: ((i + 1) % mat.cols || mat.cols) - 1,
  }
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
