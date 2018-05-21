import { loadFromJSON, predictSentence, costFunc } from '../Model'
import { rnnJSON } from './model-test-data'

// TODO: How to test with sample: true?
test('predictSentence output matches snapshot', () => {
  const lstm = loadFromJSON(rnnJSON)
  const {
    models: { model, textModel },
    hiddenSizes,
    maxCharsGen,
  } = lstm

  const predictedSentence = predictSentence({
    model,
    textModel,
    hiddenSizes,
    maxCharsGen,
    sample: false,
    temperature: 1,
  })

  expect(predictedSentence).toMatchSnapshot()
})

test('costFunc output matches snapshot', () => {
  const lstm = loadFromJSON(rnnJSON)
  const {
    models: { model, textModel },
    hiddenSizes,
  } = lstm

  const cost = costFunc({
    model,
    textModel,
    hiddenSizes,
    sentence: 'this is a test sentence',
  })

  expect(cost).toMatchSnapshot()
})

// TODO: test('loadFromJSON produces same result as create', () => {
// AND: test more variations of above two functions
