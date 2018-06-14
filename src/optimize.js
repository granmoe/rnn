import Mat from './Mat'
import { range } from './utils'

// updates weights, then resets gradients to 0; uses gradient clipping
// side-effects model
export default ({
  model,
  learningRate,
  regc,
  clipVal,
  stepCache,
  smoothEps,
  decayRate,
}) => {
  for (const [matName, mat] of Object.entries(model)) {
    if (!(matName in stepCache)) {
      stepCache[matName] = new Mat(mat.rows, mat.cols)
    }

    const s = stepCache[matName]

    for (const i of range(mat.w.length)) {
      // rmsprop adaptive learning rate
      let mdwi = mat.dw[i]
      s.w[i] = s.w[i] * decayRate + (1 - decayRate) * mdwi * mdwi

      // gradient clip
      if (Math.abs(mdwi) > clipVal) {
        mdwi = clipVal * Math.sign(mdwi)
      }

      // update (and regularize)
      mat.w[i] -= learningRate * mdwi / Math.sqrt(s.w[i] + smoothEps) - regc * mat.w[i]
      mat.dw[i] = 0 // reset gradients for next iteration
    }
  }
}
