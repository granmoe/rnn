// epochSize = sentences.length
// TODO: Show this in the UI
// $('#prepro_status').text(
// 'found ' + charList.length + ' distinct characters: ' + charList.join(''),
// )

// if (tickIter % 10 === 0) {
// draw argmax prediction
// TODO: Show this in the UI...for now just log it out
// $('#argmax').html('')
// var pred = predictSentence(model, false)
// var pred_div = '<div class="apred">' + pred + '</div>'
// $('#argmax').append(pred_div)
// // keep track of perplexity
// $('#epoch').text('epoch: ' + (tickIter / epochSize).toFixed(2))
// $('#ppl').text('perplexity: ' + cost.ppl.toFixed(2))
// $('#ticktime').text(
//   'forw/bwd time per example: ' + tickTime.toFixed(1) + 'ms',
// )
// function median(values) {
//   values.sort((a, b) => a - b) // OPT: Isn't this the default sort?
//   const half = Math.floor(values.length / 2)
//   return values.length % 2
//     ? values[half]
//     : (values[half - 1] + values[half]) / 2.0
// }
// TODO: Different solution for graph...maybe victory or something...or maybe antd has something
// if (tickIter % 100 === 0) {
// var median_ppl = median(pplList)
// pplList = []
// pplGraph.add(tickIter, median_ppl)
// pplGraph.drawSelf(document.getElementById('pplgraph'))
// }
// }

// let solverStats = solver.step(model, learningRate, regc, clipVal)
// $("#gradclip").text('grad clipped ratio: ' + solverStats.ratio_clipped)
// pplList.push(cost.ppl) // keep track of perplexity
