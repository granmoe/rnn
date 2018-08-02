const tokenize = ({ input, level = 'char' }) => {
  if (level === 'char') {
    return [...input.join('')]
  } else if (level === 'word') {
    return input.join(' ').split(/\b/)
  }
}
