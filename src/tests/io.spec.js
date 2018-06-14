import { create, loadFromJSON } from '../io'
import { predictSentence, computeCost } from '../forward'
import { rnnJSON, input } from './model-test-data'

// TODO: How to test with sample: true?
// AND: Add more variations of the two tests below
// Maybe make predictSentence and costFunc easier for caller to use, bind model args so only minimal args need to be passed, will help testing
test('predictSentence output matches snapshot', () => {
  const lstm = loadFromJSON(rnnJSON)
  const {
    models: { model, textModel },
    hiddenSizes,
  } = lstm

  const predictedSentence = predictSentence({
    model,
    textModel,
    hiddenSizes,
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

  const cost = computeCost({
    model,
    textModel,
    hiddenSizes,
    sentence: 'this is a test sentence',
  })

  expect(cost).toMatchSnapshot()
})

// TODO: Not sure how to test this...only thing I can do is call any pure funcs (like predictSentence and costFunc tests above)
test('loadFromJSON produces same result as create', () => {
  const lstmModel = create({
    type: 'lstm',
    input,
    letterSize: 5,
    hiddenSizes: [20, 20],
  })

  lstmModel.train()
  const lstmModelJSON = lstmModel.toJSON()
  const rehydratedModel = loadFromJSON(lstmModelJSON)

  expect(lstmModelJSON).toEqual(rehydratedModel.toJSON())
})
