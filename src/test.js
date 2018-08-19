const allMats = new Float32Array([...out.weights, m1.weights, m2.weights])

// how to pass in out.weights, m1.weights, m2.weights ???
// need to concat them into one huge array
// can refer to a var that has out.weights.length inside shader with ${} interpolation
// then calculate index of m1, m2, by taking normal index + length of preceding arrays
const shaderProgram = `void main(void) {
  for (int i = 0; i < ${out.weights.length}; i++) {
    const { row, col } = indexToCoord(i) // TODO

    let dot = 0
    for (int n = 0; n < ${m1.cols}; n++) {
      dot += m1.weights[n + row * m1.cols] * m2.weights[n * m2.cols + col]
    }

    commit(dot)
  }
}`

allMatsLength = out.weights.length + m1.weights.length + m2.weights.length

const glMat = turbo.alloc(allMatsLength)

glMat.data = new Float32Array(
  ...out.weights,
  ...m1.weights,
  ...m2.weights,
  ...new Float32Array(glMat.data.length - allMatsLength),
)

turbo.run(glMat, shaderProgram)
