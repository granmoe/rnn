import Mat from './Mat'

// updates weights, then resets gradients to 0; uses gradient clipping
// essentially just one function ("step") that side-effects model passed in
// TODO: could just pass in the stuff that is stored in the constructor below, then this will just be a function
// funny that there's no way for caller to control this stuff in the original code anyway
export default class Solver {
  constructor() {
    this.decayRate = 0.999
    this.smoothEps = 1e-8
    this.stepCache = {}
  }

  step(model, stepSize, regc, clipVal) {
    // perform parameter update
    let solverStats = {}
    let numClipped = 0
    let numTot = 0

    for (const [k, m] of Object.entries(model)) {
      if (!(k in this.stepCache)) {
        this.stepCache[k] = new Mat(m.rows, m.cols)
      }

      let s = this.stepCache[k]

      for (let i = 0; i < m.w.length; i++) {
        // rmsprop adaptive learning rate
        let mdwi = m.dw[i]
        s.w[i] = s.w[i] * this.decayRate + (1.0 - this.decayRate) * mdwi * mdwi

        // gradient clip
        if (mdwi > clipVal) {
          mdwi = clipVal
          numClipped++
        }
        if (mdwi < -clipVal) {
          mdwi = -clipVal
          numClipped++
        }
        numTot++

        // update (and regularize)
        m.w[i] +=
          -stepSize * mdwi / Math.sqrt(s.w[i] + this.smoothEps) - regc * m.w[i]
        m.dw[i] = 0 // reset gradients for next iteration
      }
    }

    solverStats['ratio_clipped'] = numClipped * 1.0 / numTot

    return solverStats
  }
}
