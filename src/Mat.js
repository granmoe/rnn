import { assert, randf } from './utils'

export default class Mat {
  constructor(rows, cols) {
    this.rows = rows
    this.cols = cols
    this.w = zeros(rows * cols)
    this.dw = zeros(rows * cols)

    return this
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
    this.w = this.w.map(func)
    return this
  }

  updateDw(func) {
    this.dw = this.dw.map(func)
    return this
  }

  clone({ withDw = false } = {}) {
    const copy = new Mat(this.rows, this.cols) // maybe just allow Mat constructor to take optional weights?
    copy.w = new Float64Array(this.w)
    if (withDw) {
      copy.dw = new Float64Array(this.dw)
    }
    return copy
  }

  fillRand(lo, hi) {
    return this.updateW(_ => randf(lo, hi))
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
export function RandMat(rows, cols, std) {
  return new Mat(rows, cols).fillRand(-std, std) // kind of :P
}

const zeros = count => new Float64Array(count)
