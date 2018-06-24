import Mat from './Mat'

// updates weights based on gradients and stepCache, then resets gradients to 0; uses gradient clipping
// side-effects model
export default ({
  model,
  learningRate,
  regc,
  clipVal,
  stepCache,
  smoothingEpsilon,
  decayRate,
}) => {
  for (const [matName, mat] of Object.entries(model)) {
    if (!(matName in stepCache)) {
      stepCache[matName] = new Mat(mat.rows, mat.cols)
    }

    // Perhaps store this in each mat?
    const s = stepCache[matName]

    for (let i = 0; i < mat.w.length; i++) {
      // rmsprop adaptive learning rate
      // TODO: Research and understand this
      // Maybe convert to adadelta
      let mdwi = mat.dw[i]
      // s.w[i] = prev swi * decayRate + (1 - decayRate) * dwSquared
      // s.w[i] = decayed summed squares of gradients
      s.w[i] = s.w[i] * decayRate + (1 - decayRate) * mdwi * mdwi

      /*
        RMS of grads would be:
        const rms = (nums) => Math.sqrt(nums.map(x => x * x).reduce((a, b) => a + b, 0) / nums.length)
        rms(prevGrads)
      */

      // gradient clip
      if (Math.abs(mdwi) > clipVal) {
        mdwi = clipVal * Math.sign(mdwi)
      }

      // update (and regularize)
      mat.w[i] -=
        // divisor = (decayed RMS of grads) - regc * weight
        (learningRate * mdwi) / Math.sqrt(s.w[i] + smoothingEpsilon) - regc * mat.w[i]
      mat.dw[i] = 0 // reset gradients for next iteration
    }
  }
}
