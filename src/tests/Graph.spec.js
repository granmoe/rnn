import { composeMiddlewareFunctions } from '../Graph'

describe('composeMiddlewareFunctions', () => {
  const funcs = [
    next => input => next(input + ' first'),
    next => input => next(input + ' second'),
    next => input => next(input + ' third'),
  ]

  const composedFunc = composeMiddlewareFunctions(funcs)

  const result = composedFunc('initial input')

  test('functions are invoked in order and can pass arguments to each other via next()', () => {
    expect(result).toBe('initial input first second third')
  })
})
