import { loadFromJson, predictSentence } from '../Model'
import { rnnJson } from './model-test-data'

// TODO: How to test with sample: true?
test('predictSentence output matches snapshot', () => {
  const lstm = loadFromJson(rnnJson)
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

// TODO: test('loadFromJson produces same result as create', () => {
