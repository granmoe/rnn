Before any refactoring:

TIME ELAPSED PER 500 ITERATIONS:  25180
TIME ELAPSED PER 500 ITERATIONS:  27153
TIME ELAPSED PER 500 ITERATIONS:  27035

After refactoring dot product w:








ROW = Math.ceil(i / col)
COL = i % col || col

find row and col, then do

a row * b col

dot = (_, i) => {
  const row = Math.ceil((i + 1) / c.cols)
	const col = (i + 1) % c.cols || c.cols

	let dot = 0
	for (let n = 0; n < a.cols; n++) {
		dot += a[n + ((row - 1) * a.cols)] * b[n * b.cols + col - 1]
  }

	return dot
}

const c = Array.from({ length: (a.rows * b.cols) }, dot)

O(n^3) -> O(n^2)
