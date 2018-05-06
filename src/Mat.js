import { assert, randf } from './utils'

// TODO: Change rows/cols to rowCount, colCount or numRows, numCols
export default class Mat {
  constructor(rows, cols) {
    this.rows = rows
    this.cols = cols
    this.w = zeros(rows * cols)
    this.dw = zeros(rows * cols)

    return this
  }

  indexToCoord(i) {
    return {
      row: Math.ceil((i + 1) / this.cols) - 1,
      col: ((i + 1) % this.cols || this.cols) - 1,
    }
  }

  coordToIndex(row, col) {
    // is this used anywhere? should give a once over and delete dead code everywhere
    return this.cols * row + col
  }

  get(row, col) {
    // we want row-major order
    let ix = this.cols * row + col
    assert(ix >= 0 && ix < this.w.length)
    return this.w[ix]
  }

  set(row, col, v) {
    let ix = this.cols * row + col
    assert(ix >= 0 && ix < this.w.length)
    this.w[ix] = v
  }

  updateW(func) {
    this.w.forEach((w, i) => {
      this.w[i] = func(w, i, this)
    })
    return this
  }

  updateDw(func) {
    this.dw = this.dw.map(func)
    return this
  }

  clone() {
    // does not copy over dw
    const copy = new Mat(this.rows, this.cols)
    copy.w = new Float64Array(this.w)
    return copy
  }

  toJSON() {
    let json = {}
    json['rows'] = this.rows
    json['cols'] = this.cols
    json['w'] = this.w
    return json
  }

  fromJSON(json) {
    this.rows = json.rows
    this.cols = json.cols
    this.dw = zeros(this.rows * this.cols)
    this.w = new Float64Array(json.w)
    return this
  }
}

// return Mat but filled with random numbers from gaussian
export const RandMat = (rows, cols, std) =>
  new Mat(rows, cols).updateW(_ => randf(-std, std)) // kind of :P

const zeros = count => new Float64Array(count)
