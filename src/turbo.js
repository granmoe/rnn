export default () => {
  // Mozilla reference init implementation
  const initGLFromCanvas = function(canvas) {
    const attr = { alpha: false, antialias: false }

    // Try to grab the standard context. If it fails, fallback to experimental.
    const gl =
      canvas.getContext('webgl', attr) || canvas.getContext('experimental-webgl', attr)

    // If we don't have a GL context, give up now
    if (!gl)
      throw new Error(
        'turbojs: Unable to initialize WebGL. Your browser may not support it.',
      )

    return gl
  }

  const gl = initGLFromCanvas(document.createElement('canvas'))

  // turbo.js requires a 32bit float vec4 texture. Some systems only provide 8bit/float
  // textures. A workaround is being created, but turbo.js shouldn't be used on those
  // systems anyway.
  if (!gl.getExtension('OES_texture_float'))
    throw new Error('turbojs: Required texture format OES_texture_float not supported.')

  // GPU texture buffer from JS typed array
  function newBuffer(data, DataType = Float32Array, e = gl.ARRAY_BUFFER) {
    const buf = gl.createBuffer()

    gl.bindBuffer(e, buf)
    gl.bufferData(e, new DataType(data), gl.STATIC_DRAW)

    return buf
  }

  const positionBuffer = newBuffer([-1, -1, 1, -1, 1, 1, -1, 1])
  const textureBuffer = newBuffer([0, 0, 1, 0, 1, 1, 0, 1])
  const indexBuffer = newBuffer([1, 2, 0, 3, 0, 2], Uint16Array, gl.ELEMENT_ARRAY_BUFFER)

  const vertexShaderCode =
    'attribute vec2 position;\n' +
    'varying vec2 pos;\n' +
    'attribute vec2 texture;\n' +
    '\n' +
    'void main(void) {\n' +
    '  pos = texture;\n' +
    '  gl_Position = vec4(position.xy, 0.0, 1.0);\n' +
    '}'

  const stdlib =
    '\n' +
    'precision mediump float;\n' +
    'uniform sampler2D u_texture;\n' +
    'varying vec2 pos;\n' +
    '\n' +
    'vec4 read(void) {\n' +
    '  return texture2D(u_texture, pos);\n' +
    '}\n' +
    '\n' +
    'vec4 readIndex(vec2 index) {\n' +
    // TODO: Convert to vec2 equivalent
    '  return texture2D(u_texture, index);\n' +
    '}\n' +
    '\n' +
    'void commit(vec4 val) {\n' +
    '  gl_FragColor = val;\n' +
    '}\n' +
    '\n' +
    '// user code begins here\n' +
    '\n'

  const vertexShader = gl.createShader(gl.VERTEX_SHADER)

  gl.shaderSource(vertexShader, vertexShaderCode)
  gl.compileShader(vertexShader)

  // This should not fail.
  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
    throw new Error(
      '\nturbojs: Could not build internal vertex shader (fatal).\n' +
        '\n' +
        "INFO: >REPORT< THIS. That's our fault!\n" +
        '\n' +
        '--- CODE DUMP ---\n' +
        vertexShaderCode +
        '\n\n' +
        '--- ERROR LOG ---\n' +
        gl.getShaderInfoLog(vertexShader),
    )

  // Transfer data onto clamped texture and turn off any filtering
  function createTexture(data, size) {
    const texture = gl.createTexture()

    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.FLOAT, data)
    gl.bindTexture(gl.TEXTURE_2D, null)

    return texture
  }

  return {
    // run code against a pre-allocated array
    run: function(ipt, code) {
      const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)

      gl.shaderSource(fragmentShader, stdlib + code)

      gl.compileShader(fragmentShader)

      // Use this output to debug the shader
      // Keep in mind that WebGL GLSL is **much** stricter than e.g. OpenGL GLSL
      if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
        const LOC = code.split('\n')
        let dbgMsg =
          'ERROR: Could not build shader (fatal).\n\n------------------ KERNEL CODE DUMP ------------------\n'

        for (let nl = 0; nl < LOC.length; nl++)
          dbgMsg += stdlib.split('\n').length + nl + '> ' + LOC[nl] + '\n'

        dbgMsg +=
          '\n--------------------- ERROR  LOG ---------------------\n' +
          gl.getShaderInfoLog(fragmentShader)

        throw new Error(dbgMsg)
      }

      const program = gl.createProgram()

      gl.attachShader(program, vertexShader)
      gl.attachShader(program, fragmentShader)
      gl.linkProgram(program)

      if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        throw new Error('turbojs: Failed to link GLSL program code.')

      const uTexture = gl.getUniformLocation(program, 'u_texture')
      const aPosition = gl.getAttribLocation(program, 'position')
      const aTexture = gl.getAttribLocation(program, 'texture')

      gl.useProgram(program)

      const size = Math.sqrt(ipt.data.length) / 4
      const texture = createTexture(ipt.data, size)

      gl.viewport(0, 0, size, size)
      gl.bindFramebuffer(gl.FRAMEBUFFER, gl.createFramebuffer())

      // Types arrays speed this up tremendously.
      const nTexture = createTexture(new Float32Array(ipt.data.length), size)

      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        nTexture,
        0,
      )

      // Test for mobile bug MDN->WebGL_best_practices, bullet 7
      const frameBufferStatus =
        gl.checkFramebufferStatus(gl.FRAMEBUFFER) == gl.FRAMEBUFFER_COMPLETE // eslint-disable-line

      if (!frameBufferStatus)
        throw new Error(
          'turbojs: Error attaching float texture to framebuffer. Your device is probably incompatible. Error info: ' +
            frameBufferStatus.message,
        )

      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.activeTexture(gl.TEXTURE0)
      gl.uniform1i(uTexture, 0)
      gl.bindBuffer(gl.ARRAY_BUFFER, textureBuffer)
      gl.enableVertexAttribArray(aTexture)
      gl.vertexAttribPointer(aTexture, 2, gl.FLOAT, false, 0, 0)
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
      gl.enableVertexAttribArray(aPosition)
      gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0)
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer)
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0)
      gl.readPixels(0, 0, size, size, gl.RGBA, gl.FLOAT, ipt.data)
      //                                 ^ 4 x 32 bit ^

      return ipt.data.subarray(0, ipt.length)
    },
    alloc: function(sz) {
      // A sane limit for most GPUs out there.
      // JS falls apart before GLSL limits could ever be reached.
      if (sz > 16777216)
        throw new Error('turbojs: Whoops, the maximum array size is exceeded!')

      var ns = Math.pow(Math.pow(2, Math.ceil(Math.log(sz) / 1.386) - 1), 2)
      return {
        data: new Float32Array(ns * 16),
        length: sz,
      }
    },
  }
}
