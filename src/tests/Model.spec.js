import { create, loadFromJson } from '../Model'
import { rnnJson, input } from './model-test-data'

test('train function output matches snapshot', done => {
  const { train, pause } = loadFromJson(rnnJson)
  train(result => {
    expect(result.argMaxPrediction).toMatchSnapshot()
    pause()
    done()
  }, 1)
})

test('loadFromJson produces same result as create', done => {
  const originalLSTM = create({
    type: 'lstm',
    input,
    letterSize: 5,
    hiddenSizes: [20, 20],
  })

  const json = originalLSTM.toJSON()
  const copiedLSTM = loadFromJson(json)

  originalLSTM.train(result => {
    const originalLSTMResult = result.argMaxPrediction
    originalLSTM.pause()

    copiedLSTM.train(result => {
      const copiedLSTMResult = result.argMaxPrediction
      copiedLSTM.pause()
      console.log(originalLSTMResult, copiedLSTMResult)
      expect(originalLSTMResult).toEqual(copiedLSTMResult)
      done()
    }, 1)
  }, 1)
})
