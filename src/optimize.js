// updates weights based on gradients and cachedGradients, then resets gradients to 0
// uses gradient clipping
export default ({ graph, learningRate, regc, clipVal, smoothingEpsilon, decayRate }) => {
  for (let layer of graph.layers) {
    // TODO: Graph can provide an optimize method that calls this function, passing in itself?
    const { cachedGradients, gradients, weights } = layer

    for (let i = 0; i < layer.length; i++) {
      // rmsprop adaptive learning rate
      // TODO: Research and understand this, maybe convert to adadelta or another optimization method
      let gradient = gradients[i]
      // cachedGradients[i] = prev swi * decayRate + (1 - decayRate) * gradientSquared
      //                    = decayed summed squares of gradients
      cachedGradients[i] =
        cachedGradients[i] * decayRate + (1 - decayRate) * gradient * gradient

      /*
        RMS of grads would be:
        const rms = (nums) => Math.sqrt(nums.map(x => x * x).reduce((a, b) => a + b, 0) / nums.length)
        rms(prevGrads)
      */

      // gradient clip
      if (Math.abs(gradient) > clipVal) {
        gradients[i] = clipVal * Math.sign(gradient)
      }

      // update (and regularize)
      weights[i] -=
        // divisor = (decayed RMS of grads) - regc * weight
        (learningRate * gradient) / Math.sqrt(cachedGradients[i] + smoothingEpsilon) -
        regc * weights[i]
    }

    layer.resetGradients()
  }
}
