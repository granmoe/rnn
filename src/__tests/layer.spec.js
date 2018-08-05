import { cloneMat, createRandomLayer } from '../layer'

describe('createRandomLayer', () => {
  test('produces a matrix of the size given in the args with random values within the bounds given by the third arg', () => {
    const result = createRandomLayer(20, 30, 0.5)

    expect(result.rows).toBe(20)
    expect(result.cols).toBe(30)

    result.weights.forEach(weight => {
      expect(weight < 0.5 && weight > -0.5).toBeTruthy()
    })
  })

  test('produces a with random values between -0.08 and 0.08 when no bound is provided', () => {
    const result = createRandomLayer(40, 1)

    result.weights.forEach(weight => {
      expect(weight < 0.08 && weight > -0.08).toBeTruthy()
    })
  })
})

describe('cloneMat', () => {
  test('returns a new matrix with the same weights as the input matrix, but zeroed gradients', () => {
    const mat = createRandomLayer(10, 10)
    const clonedMat = cloneMat(mat)

    clonedMat.weights.forEach((weight, index) => {
      expect(weight).toEqual(mat.weights[index])
    })

    clonedMat.gradients.forEach(gradient => {
      expect(gradient).toEqual(0)
    })
  })
})
