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

    // TODO: Graph can provide an optimize method that calls this function, passing in itself?
    // Graph needs a reference to each mat...maybe just throw each one in an array
    // store this in each mat
    const s = stepCache[matName]

    for (let i = 0; i < mat.weights.length; i++) {
      // rmsprop adaptive learning rate
      // TODO: Research and understand this
      // Maybe convert to adadelta
      let mGradienti = mat.gradients[i]
      // s.weights[i] = prev swi * decayRate + (1 - decayRate) * gradientSquared
      // s.weights[i] = decayed summed squares of gradients
      s.weights[i] = s.weights[i] * decayRate + (1 - decayRate) * mGradienti * mGradienti

      /*
        RMS of grads would be:
        const rms = (nums) => Math.sqrt(nums.map(x => x * x).reduce((a, b) => a + b, 0) / nums.length)
        rms(prevGrads)
      */

      // gradient clip
      if (Math.abs(mGradienti) > clipVal) {
        mGradienti = clipVal * Math.sign(mGradienti)
      }

      // update (and regularize)
      mat.weights[i] -=
        // divisor = (decayed RMS of grads) - regc * weight
        (learningRate * mGradienti) / Math.sqrt(s.weights[i] + smoothingEpsilon) -
        regc * mat.weights[i]
      mat.gradients[i] = 0 // reset gradients for next iteration
    }
  }
}
