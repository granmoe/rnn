const dt = require('dependency-tree')

const tree = dt({
  filename: 'index.js',
  directory: 'Users/mattgranmoe/code/ai/rnn/src',
  filter: path => path.indexOf('node_modules') === -1, // optional
  // nonExistent: [], // optional
})

const removeBasePath = str =>
  str.replace('/Users/mattgranmoe/code/ai/rnn/src/', '').replace('.js', '')

const removeBasePathFromKeys = obj => {
  if (Object.keys(obj).length) {
    return Object.entries(obj).reduce(
      (result, [k, v]) => ({
        ...result,
        [removeBasePath(k)]: removeBasePathFromKeys(v),
      }),
      {},
    )
  }

  return obj
}

const trimmedTree = removeBasePathFromKeys(tree)

console.log('\n', trimmedTree)
console.log('\n', JSON.stringify(trimmedTree), '\n')
