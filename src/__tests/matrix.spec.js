import createMat, { createRandomMat } from '../matrix'

describe('createRandomMat', () => {
  test('produces a matrix of the size given in the args with random values within the bounds given by the third arg', () => {
    const result = createRandomMat(20, 30, 0.5)

    expect(result.rows).toBe(20)
    expect(result.cols).toBe(30)

    result.weights.forEach(weight => {
      expect(weight < 0.5 && weight > -0.5).toBeTruthy()
    })
  })

  test('produces a with random values between -0.08 and 0.08 when no bound is provided', () => {
    const result = createRandomMat(40, 1)

    result.weights.forEach(weight => {
      expect(weight < 0.08 && weight > -0.08).toBeTruthy()
    })
  })
})

describe('clone', () => {
  test('returns a new matrix with the same weights as the input matrix, but zeroed gradients', () => {
    const mat = createRandomMat(10, 10)
    const clonedMat = mat.clone()

    clonedMat.weights.forEach((weight, index) => {
      expect(weight).toEqual(mat.weights[index])
    })

    clonedMat.gradients.forEach(gradient => {
      expect(gradient).toEqual(0)
    })
  })
})

describe('indexToCoord', () => {
  const mat = createMat(10, 10)

  test('returns an object { row, col } for a given matrix and index', () => {
    expect(mat.indexToCoord(0)).toEqual({ row: 0, col: 0 })
    expect(mat.indexToCoord(10)).toEqual({ row: 1, col: 0 })
    expect(mat.indexToCoord(11)).toEqual({ row: 1, col: 1 })
    expect(mat.indexToCoord(98)).toEqual({ row: 9, col: 8 })
    expect(mat.indexToCoord(99)).toEqual({ row: 9, col: 9 })
  })

  test('throws error when passing index > matrix.length', () => {
    expect(() => mat.indexToCoord(100)).toThrowError()
  })
})

describe('updateGradients', () => {
  test('uses passed in function to update gradient values of a matrix', () => {
    const mat = createMat(2, 2)
    const expectedGradients = [0, 1, 2, 3]

    mat.updateGradients((grad, i) => grad + i)
    mat.gradients.forEach((grad, i) => {
      expect(grad).toBe(expectedGradients[i])
    })
  })
})

describe('updateWeights', () => {
  test('uses passed in function to update weights of a matrix', () => {
    const mat = createMat(2, 2)
    const expectedWeights = [0, 1, 2, 3]

    mat.updateWeights((weight, i) => weight + i)
    mat.weights.forEach((weight, i) => {
      expect(weight).toBe(expectedWeights[i])
    })
  })
})
