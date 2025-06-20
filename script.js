'use strict';

// Get the canvas element by its ID
const canvas = document.getElementById('fluid');

// Initial resize of the canvas to match its display size
resizeCanvas();

// Configuration object for the fluid simulation parameters
let config = {
    SIM_RESOLUTION: 128,      // Resolution of the simulation grid for velocity and pressure
    DYE_RESOLUTION: 1440,     // Resolution of the dye (color) grid
    CAPTURE_RESOLUTION: 512,  // Resolution for capturing screenshots (not actively used in this base code)
    DENSITY_DISSIPATION: 3.5, // How quickly dye dissipates
    VELOCITY_DISSIPATION: 2,  // How quickly velocity dissipates
    PRESSURE: 0.1,            // Amount of pressure applied (related to density)
    PRESSURE_ITERATIONS: 20,  // Number of iterations for pressure solver (higher = more accurate)
    CURL: 10,                 // Strength of vorticity confinement (adds swirls)
    SPLAT_RADIUS: 0.5,        // Size of the splat (ink drop) when interacting
    SPLAT_FORCE: 6000,        // Force applied by a splat
    SHADING: true,            // Enable/disable shading for depth perception
    COLOR_UPDATE_SPEED: 10,   // Speed at which colors change over time
    PAUSED: false,            // Simulation pause state (not actively used for control here)
    BACK_COLOR: { r: 0, g: 0, b: 0 }, // Background color (not directly used by GL clear here due to alpha)
    TRANSPARENT: true,        // Enable/disable transparency (used in GL context params)
};

/**
 * Defines the properties for a pointer (mouse or touch) interaction.
 */
function pointerPrototype() {
    this.id = -1; // Unique identifier for the pointer
    this.texcoordX = 0; // Current X texture coordinate
    this.texcoordY = 0; // Current Y texture coordinate
    this.prevTexcoordX = 0; // Previous X texture coordinate
    this.prevTexcoordY = 0; // Previous Y texture coordinate
    this.deltaX = 0; // Change in X position
    this.deltaY = 0; // Change in Y position
    this.down = false; // Is the pointer currently pressed down
    this.moved = false; // Has the pointer moved since last update
    this.color = [30, 0, 300]; // Initial color for the splat
}

let pointers = [];
pointers.push(new pointerPrototype()); // Add the initial pointer for mouse

// Get the WebGL rendering context and extensions
const { gl, ext } = getWebGLContext(canvas);

// Adjust dye resolution and shading if linear filtering is not supported
if (!ext.supportLinearFiltering) {
    config.DYE_RESOLUTION = 512;
    config.SHADING = false;
}

/**
 * Initializes and returns the WebGL rendering context.
 * It tries to get WebGL2, falls back to WebGL1, and checks for necessary extensions.
 * @param {HTMLCanvasElement} canvas The canvas element.
 * @returns {{gl: WebGLRenderingContext|WebGL2RenderingContext, ext: object}} The GL context and supported extensions.
 */
function getWebGLContext(canvas) {
    // WebGL context parameters, alpha:true allows transparency
    const params = { alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false };

    let gl = canvas.getContext('webgl2', params);
    const isWebGL2 = !!gl;
    if (!isWebGL2) {
        gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);
    }

    let halfFloat;
    let supportLinearFiltering;
    if (isWebGL2) {
        gl.getExtension('EXT_color_buffer_float'); // Needed for float textures as render targets
        supportLinearFiltering = gl.getExtension('OES_texture_float_linear'); // Needed for linear filtering on float textures
    } else {
        halfFloat = gl.getExtension('OES_texture_half_float'); // Needed for half-float textures in WebGL1
        supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear'); // Needed for linear filtering on half-float textures
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0); // Set clear color (transparent black in this case)

    const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
    let formatRGBA;
    let formatRG;
    let formatR;

    // Determine supported texture formats based on WebGL version
    if (isWebGL2) {
        formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
    } else {
        // WebGL1 fallback: use RGBA for all formats if specific ones are not available
        formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    }

    return {
        gl,
        ext: {
            formatRGBA,
            formatRG,
            formatR,
            halfFloatTexType,
            supportLinearFiltering
        }
    };
}

/**
 * Checks if a texture format is supported for rendering and returns the best available format.
 * @param {WebGLRenderingContext|WebGL2RenderingContext} gl The WebGL context.
 * @param {number} internalFormat The desired internal format.
 * @param {number} format The desired format.
 * @param {number} type The desired data type.
 * @returns {{internalFormat: number, format: number}|null} The supported format or null.
 */
function getSupportedFormat(gl, internalFormat, format, type) {
    if (!supportRenderTextureFormat(gl, internalFormat, format, type)) {
        // Fallback to less specific formats if the ideal one isn't supported
        switch (internalFormat) {
            case gl.R16F:
                return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
            case gl.RG16F:
                return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
            default:
                return null;
        }
    }
    return {
        internalFormat,
        format
    }
}

/**
 * Checks if a texture format can be used as a render target in a framebuffer.
 * @param {WebGLRenderingContext|WebGL2RenderingContext} gl The WebGL context.
 * @param {number} internalFormat The internal format of the texture.
 * @param {number} format The format of the texture.
 * @param {number} type The data type of the texture.
 * @returns {boolean} True if renderable, false otherwise.
 */
function supportRenderTextureFormat(gl, internalFormat, format, type) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    return status === gl.FRAMEBUFFER_COMPLETE;
}

/**
 * Represents a WebGL material, managing multiple shader programs with keywords.
 */
class Material {
    /**
     * @param {WebGLShader} vertexShader The shared vertex shader.
     * @param {string} fragmentShaderSource The base fragment shader source string.
     */
    constructor(vertexShader, fragmentShaderSource) {
        this.vertexShader = vertexShader;
        this.fragmentShaderSource = fragmentShaderSource;
        this.programs = []; // Cache for compiled programs based on keywords
        this.activeProgram = null;
        this.uniforms = [];
    }

    /**
     * Sets active keywords and compiles/uses the appropriate shader program.
     * @param {string[]} keywords Array of keywords to activate in the fragment shader.
     */
    setKeywords(keywords) {
        let hash = 0;
        // Generate a hash based on keywords for caching programs
        for (let i = 0; i < keywords.length; i++) {
            hash += hashCode(keywords[i]);
        }

        let program = this.programs[hash];
        if (program == null) {
            // If program not cached, compile it
            let fragmentShader = compileShader(gl.FRAGMENT_SHADER, this.fragmentShaderSource, keywords);
            program = createProgram(this.vertexShader, fragmentShader);
            this.programs[hash] = program;
        }

        if (program === this.activeProgram) return; // No change needed

        this.uniforms = getUniforms(program); // Get uniforms for the new program
        this.activeProgram = program;
    }

    /**
     * Binds the currently active shader program.
     */
    bind() {
        gl.useProgram(this.activeProgram);
    }
}

/**
 * Represents a basic WebGL shader program.
 */
class Program {
    /**
     * @param {WebGLShader} vertexShader The vertex shader.
     * @param {WebGLShader} fragmentShader The fragment shader.
     */
    constructor(vertexShader, fragmentShader) {
        this.uniforms = {};
        this.program = createProgram(vertexShader, fragmentShader);
        this.uniforms = getUniforms(this.program);
    }

    /**
     * Binds the shader program.
     */
    bind() {
        gl.useProgram(this.program);
    }
}

/**
 * Creates and links a WebGL program from vertex and fragment shaders.
 * @param {WebGLShader} vertexShader The compiled vertex shader.
 * @param {WebGLShader} fragmentShader The compiled fragment shader.
 * @returns {WebGLProgram} The linked WebGL program.
 */
function createProgram(vertexShader, fragmentShader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.trace(gl.getProgramInfoLog(program));
    }

    return program;
}

/**
 * Retrieves all active uniform locations for a given WebGL program.
 * @param {WebGLProgram} program The WebGL program.
 * @returns {object} An object mapping uniform names to their locations.
 */
function getUniforms(program) {
    let uniforms = {};
    let uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
        let uniformName = gl.getActiveUniform(program, i).name;
        uniforms[uniformName] = gl.getUniformLocation(program, uniformName);
    }
    return uniforms;
}

/**
 * Compiles a WebGL shader.
 * @param {number} type The type of shader (gl.VERTEX_SHADER or gl.FRAGMENT_SHADER).
 * @param {string} source The GLSL source code.
 * @param {string[]} [keywords] Optional array of keywords to define in the shader.
 * @returns {WebGLShader} The compiled shader.
 */
function compileShader(type, source, keywords) {
    source = addKeywords(source, keywords); // Add #define directives based on keywords

    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.trace(gl.getShaderInfoLog(shader));
    }

    return shader;
}

/**
 * Adds #define directives to a shader source based on keywords.
 * @param {string} source The original shader source.
 * @param {string[]} keywords Array of keywords.
 * @returns {string} The modified shader source.
 */
function addKeywords(source, keywords) {
    if (keywords == null) return source;
    let keywordsString = '';
    keywords.forEach(keyword => {
        keywordsString += '#define ' + keyword + '\n';
    });
    return keywordsString + source;
}

// --- GLSL Shader Sources (compiled here) ---

// Vertex shader common to most passes
const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
  precision highp float;

  attribute vec2 aPosition;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform vec2 texelSize;

  void main () {
      vUv = aPosition * 0.5 + 0.5;
      vL = vUv - vec2(texelSize.x, 0.0);
      vR = vUv + vec2(texelSize.x, 0.0);
      vT = vUv + vec2(0.0, texelSize.y);
      vB = vUv - vec2(0.0, texelSize.y);
      gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`);

// Vertex shader for blur operations
const blurVertexShader = compileShader(gl.VERTEX_SHADER, `
  precision highp float;

  attribute vec2 aPosition;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  uniform vec2 texelSize;

  void main () {
      vUv = aPosition * 0.5 + 0.5;
      float offset = 1.33333333; // Offset for samples
      vL = vUv - texelSize * offset; // Left sample coordinate
      vR = vUv + texelSize * offset; // Right sample coordinate
      gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`);

// Fragment shader for blurring (horizontal or vertical)
const blurShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  precision mediump sampler2D;

  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  uniform sampler2D uTexture;

  void main () {
      // Weighted average of current and two offset samples
      vec4 sum = texture2D(uTexture, vUv) * 0.29411764;
      sum += texture2D(uTexture, vL) * 0.35294117;
      sum += texture2D(uTexture, vR) * 0.35294117;
      gl_FragColor = sum;
  }
`);

// Fragment shader for copying a texture
const copyShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  precision mediump sampler2D;

  varying highp vec2 vUv;
  uniform sampler2D uTexture;

  void main () {
      gl_FragColor = texture2D(uTexture, vUv);
  }
`);

// Fragment shader for clearing a texture with a weighted value
const clearShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  precision mediump sampler2D;

  varying highp vec2 vUv;
  uniform sampler2D uTexture;
  uniform float value; // Clears by multiplying existing texture with this value

  void main () {
      gl_FragColor = value * texture2D(uTexture, vUv);
  }
`);

// Fragment shader for rendering a solid color
const colorShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;

  uniform vec4 color; // Uniform for the color to render

  void main () {
      gl_FragColor = color;
  }
`);

// Fragment shader source for displaying the fluid simulation
// Includes optional shading based on density gradients
const displayShaderSource = `
  precision highp float;
  precision highp sampler2D;

  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uTexture; // The fluid density texture
  uniform sampler2D uDithering; // Dithering texture for visual quality
  uniform vec2 ditherScale; // Scale for dithering texture
  uniform vec2 texelSize; // Size of a texel in UV space

  // Converts linear color space to gamma color space for display
  vec3 linearToGamma (vec3 color) {
      color = max(color, vec3(0));
      return max(1.055 * pow(color, vec3(0.416666667)) - 0.055, vec3(0));
  }

  void main () {
      vec3 c = texture2D(uTexture, vUv).rgb; // Current pixel's color from density texture

  #ifdef SHADING // Apply shading if SHADING keyword is defined
      vec3 lc = texture2D(uTexture, vL).rgb; // Left sample
      vec3 rc = texture2D(uTexture, vR).rgb; // Right sample
      vec3 tc = texture2D(uTexture, vT).rgb; // Top sample
      vec3 bc = texture2D(uTexture, vB).rgb; // Bottom sample

      float dx = length(rc) - length(lc); // Horizontal gradient of luminance
      float dy = length(tc) - length(bc); // Vertical gradient of luminance

      vec3 n = normalize(vec3(dx, dy, length(texelSize))); // Surface normal vector
      vec3 l = vec3(0.0, 0.0, 1.0); // Light direction (from front)

      float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0); // Diffuse lighting calculation
      c *= diffuse; // Apply lighting to color
  #endif

      // Use the maximum component for alpha, ensuring transparency where fluid is absent
      float a = max(c.r, max(c.g, c.b));
      gl_FragColor = vec4(c, a); // Final color output
  }
`;

// Fragment shader for applying splats (velocity and density impulses)
const splatShader = compileShader(gl.FRAGMENT_SHADER, `
  precision highp float;
  precision highp sampler2D;

  varying vec2 vUv;
  uniform sampler2D uTarget; // The texture to splat onto (velocity or dye)
  uniform float aspectRatio; // Aspect ratio of the canvas
  uniform vec3 color; // Color/velocity impulse to splat
  uniform vec2 point; // Center of the splat in normalized UV coordinates
  uniform float radius; // Radius of the splat

  void main () {
      vec2 p = vUv - point.xy; // Vector from splat center to current pixel
      p.x *= aspectRatio; // Adjust for aspect ratio
      // Gaussian falloff for the splat
      vec3 splat = exp(-dot(p, p) / radius) * color;
      vec3 base = texture2D(uTarget, vUv).xyz; // Existing value from target texture
      gl_FragColor = vec4(base + splat, 1.0); // Add splat to existing value
  }
`);

// Fragment shader for advection (moving fluid properties with velocity)
const advectionShader = compileShader(gl.FRAGMENT_SHADER, `
  precision highp float;
  precision highp sampler2D;

  varying vec2 vUv;
  uniform sampler2D uVelocity; // Velocity field
  uniform sampler2D uSource;   // Texture to advect (e.g., dye, velocity itself)
  uniform vec2 texelSize;     // Texel size of the velocity texture
  uniform vec2 dyeTexelSize;  // Texel size of the dye texture (only if MANUAL_FILTERING)
  uniform float dt;            // Delta time
  uniform float dissipation;   // Dissipation factor

  // Manual bilinear interpolation function (for WebGL1 or when linear filtering is not supported)
  vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
      vec2 st = uv / tsize - 0.5; // Convert UV to grid space
      vec2 iuv = floor(st);       // Integer part of grid coordinates
      vec2 fuv = fract(st);       // Fractional part

      // Sample 4 nearest texels
      vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
      vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
      vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
      vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);

      // Bilinear interpolation
      return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
  }

  void main () {
  #ifdef MANUAL_FILTERING
      // Calculate previous position using velocity and manual filtering
      vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
      vec4 result = bilerp(uSource, coord, dyeTexelSize);
  #else
      // Calculate previous position using velocity and hardware linear filtering
      vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
      vec4 result = texture2D(uSource, coord);
  #endif
      float decay = 1.0 + dissipation * dt; // Calculate decay factor
      gl_FragColor = result / decay; // Apply advection and dissipation
  }
`,
    ext.supportLinearFiltering ? null : ['MANUAL_FILTERING'] // Add MANUAL_FILTERING keyword if linear filtering is not supported
);

// Fragment shader for calculating divergence of the velocity field
const divergenceShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  precision mediump sampler2D;

  varying highp vec2 vUv;
  varying highp vec2 vL; // Left neighbor UV
  varying highp vec2 vR; // Right neighbor UV
  varying highp vec2 vT; // Top neighbor UV
  varying highp vec2 vB; // Bottom neighbor UV
  uniform sampler2D uVelocity; // Velocity field texture

  void main () {
      // Sample velocity from neighbors
      float L = texture2D(uVelocity, vL).x;
      float R = texture2D(uVelocity, vR).x;
      float T = texture2D(uVelocity, vT).y;
      float B = texture2D(uVelocity, vB).y;

      vec2 C = texture2D(uVelocity, vUv).xy; // Current cell's velocity
      // Apply boundary conditions: reflect velocity at edges
      if (vL.x < 0.0) { L = -C.x; }
      if (vR.x > 1.0) { R = -C.x; }
      if (vT.y > 1.0) { T = -C.y; }
      if (vB.y < 0.0) { B = -C.y; }

      // Calculate divergence (how much fluid is expanding/contracting)
      float div = 0.5 * (R - L + T - B);
      gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
  }
`);

// Fragment shader for calculating curl (vorticity) of the velocity field
const curlShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  precision mediump sampler2D;

  varying highp vec2 vUv;
  varying highp vec2 vL;
  varying highp vec2 vR;
  varying highp vec2 vT;
  varying highp vec2 vB;
  uniform sampler2D uVelocity; // Velocity field texture

  void main () {
      // Sample velocity components from neighbors to calculate cross-derivative
      float L = texture2D(uVelocity, vL).y; // Vy at left
      float R = texture2D(uVelocity, vR).y; // Vy at right
      float T = texture2D(uVelocity, vT).x; // Vx at top
      float B = texture2D(uVelocity, vB).x; // Vx at bottom
      // Calculate vorticity (tendency to rotate)
      float vorticity = R - L - T + B;
      gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
  }
`);

// Fragment shader for applying vorticity confinement force
const vorticityShader = compileShader(gl.FRAGMENT_SHADER, `
  precision highp float;
  precision highp sampler2D;

  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity; // Velocity field
  uniform sampler2D uCurl;     // Curl (vorticity) texture
  uniform float curl;          // Curl strength
  uniform float dt;            // Delta time

  void main () {
      // Sample curl values from neighbors
      float L = texture2D(uCurl, vL).x;
      float R = texture2D(uCurl, vR).x;
      float T = texture2D(uCurl, vT).x;
      float B = texture2D(uCurl, vB).x;
      float C = texture2D(uCurl, vUv).x; // Current curl value

      // Calculate force direction based on curl gradient
      vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
      force /= length(force) + 0.0001; // Normalize and prevent division by zero
      force *= curl * C; // Scale by curl strength and current curl value
      force.y *= -1.0; // Invert Y-component for consistent direction

      vec2 velocity = texture2D(uVelocity, vUv).xy; // Current velocity
      velocity += force * dt; // Add vorticity force to velocity
      velocity = min(max(velocity, -1000.0), 1000.0); // Clamp velocity to prevent extreme values
      gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`);

// Fragment shader for solving pressure using Jacobi iterations
const pressureShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  precision mediump sampler2D;

  varying highp vec2 vUv;
  varying highp vec2 vL;
  varying highp vec2 vR;
  varying highp vec2 vT;
  varying highp vec2 vB;
  uniform sampler2D uPressure;    // Current pressure field
  uniform sampler2D uDivergence;  // Divergence field

  void main () {
      // Sample pressure from neighbors
      float L = texture2D(uPressure, vL).x;
      float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x;
      float B = texture2D(uPressure, vB).x;
      float C = texture2D(uPressure, vUv).x; // Current cell's pressure (for reference, though not directly used in weighted average)
      float divergence = texture2D(uDivergence, vUv).x; // Divergence at current cell
      // Calculate new pressure: average of neighbors minus divergence (Jacobi iteration)
      float pressure = (L + R + B + T - divergence) * 0.25;
      gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
  }
`);

// Fragment shader for subtracting the pressure gradient from the velocity field
const gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  precision mediump sampler2D;

  varying highp vec2 vUv;
  varying highp vec2 vL;
  varying highp vec2 vR;
  varying highp vec2 vT;
  varying highp vec2 vB;
  uniform sampler2D uPressure; // Pressure field
  uniform sampler2D uVelocity; // Velocity field

  void main () {
      // Sample pressure from neighbors
      float L = texture2D(uPressure, vL).x;
      float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x;
      float B = texture2D(uPressure, vB).x;
      vec2 velocity = texture2D(uVelocity, vUv).xy; // Current velocity
      // Subtract pressure gradient (R-L for X, T-B for Y)
      velocity.xy -= vec2(R - L, T - B);
      gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`);

/**
 * Utility function to render a full-screen quad.
 * This is used for all simulation passes, drawing the result of a shader to a target framebuffer.
 * @param {object} target The target FBO (Framebuffer Object) to render to, or null for the default canvas framebuffer.
 * @param {boolean} [clear=false] Whether to clear the target framebuffer before drawing.
 */
const blit = (() => {
    // Set up vertex buffer for a quad that covers the entire screen (-1 to 1 in normalized device coordinates)
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    // Set up index buffer for drawing two triangles to form the quad
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    // Specify layout for vertex attribute (position at index 0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0); // Enable the vertex attribute

    return (target, clear = false) => {
        if (target == null) {
            // Render to the default canvas framebuffer
            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        } else {
            // Render to a specified framebuffer object
            gl.viewport(0, 0, target.width, target.height);
            gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        }
        if (clear) {
            gl.clearColor(0.0, 0.0, 0.0, 1.0); // Set clear color (black)
            gl.clear(gl.COLOR_BUFFER_BIT); // Clear the color buffer
        }
        // CHECK_FRAMEBUFFER_STATUS(); // Uncomment for debugging framebuffer status
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0); // Draw the quad (2 triangles)
    }
})();

// Framebuffer status check function (commented out by default, uncomment for debugging)
/*
function CHECK_FRAMEBUFFER_STATUS() {
    let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        console.trace("Framebuffer error: " + status);
    }
}
*/

let dye;        // Double FBO for dye (color) density
let velocity;   // Double FBO for velocity field
let divergence; // Single FBO for divergence field
let curl;       // Single FBO for curl (vorticity) field
let pressure;   // Double FBO for pressure field

// Load a dithering texture (used in displayShaderSource if SHADING is enabled)
// Using a placeholder URL as the original path is theme-specific.
let ditheringTexture = createTextureAsync('https://placehold.co/1x1/000000/FFFFFF?text=');

// Initialize all GL programs
const blurProgram = new Program(blurVertexShader, blurShader);
const copyProgram = new Program(baseVertexShader, copyShader);
const clearProgram = new Program(baseVertexShader, clearShader);
const colorProgram = new Program(baseVertexShader, colorShader);
const splatProgram = new Program(baseVertexShader, splatShader);
const advectionProgram = new Program(baseVertexShader, advectionShader);
const divergenceProgram = new Program(baseVertexShader, divergenceShader);
const curlProgram = new Program(baseVertexShader, curlShader);
const vorticityProgram = new Program(baseVertexShader, vorticityShader);
const pressureProgram = new Program(baseVertexShader, pressureShader);
const gradienSubtractProgram = new Program(baseVertexShader, gradientSubtractShader);

// Initialize display material (handles shading based on config)
const displayMaterial = new Material(baseVertexShader, displayShaderSource);

/**
 * Initializes or resizes all framebuffers (FBOs) used in the simulation.
 * This is called on initial load and when the canvas is resized.
 */
function initFramebuffers() {
    let simRes = getResolution(config.SIM_RESOLUTION); // Resolution for velocity/pressure
    let dyeRes = getResolution(config.DYE_RESOLUTION); // Resolution for dye

    const texType = ext.halfFloatTexType; // Half-float texture type
    const rgba = ext.formatRGBA; // RGBA format
    const rg = ext.formatRG;     // RG (red-green) format for velocity
    const r = ext.formatR;       // R (red) format for scalar fields like divergence/pressure
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST; // Linear or nearest filtering

    gl.disable(gl.BLEND); // Disable blending for simulation passes

    // Initialize or resize dye FBO
    if (dye == null)
        dye = createDoubleFBO(dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering);
    else
        dye = resizeDoubleFBO(dye, dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filtering);

    // Initialize or resize velocity FBO
    if (velocity == null)
        velocity = createDoubleFBO(simRes.width, simRes.height, rg.internalFormat, rg.format, texType, filtering);
    else
        velocity = resizeDoubleFBO(velocity, simRes.width, simRes.height, rg.internalFormat, rg.format, texType, filtering);

    // Initialize or resize single FBOs
    divergence = createFBO(simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
    curl = createFBO(simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
    pressure = createDoubleFBO(simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
}

/**
 * Creates a single Framebuffer Object (FBO) with a texture attachment.
 * @param {number} w Width of the texture.
 * @param {number} h Height of the texture.
 * @param {number} internalFormat Internal format of the texture.
 * @param {number} format Format of the texture.
 * @param {number} type Data type of the texture.
 * @param {number} param Texture filtering parameter (e.g., gl.LINEAR).
 * @returns {object} FBO object containing texture, fbo, dimensions, texel size, and attach method.
 */
function createFBO(w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0); // Activate texture unit 0
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture); // Bind the new texture
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param); // Minification filter
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param); // Magnification filter
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); // Clamp to edge for S coordinate
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); // Clamp to edge for T coordinate
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null); // Allocate texture memory

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); // Bind the new framebuffer
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); // Attach texture to FBO
    gl.viewport(0, 0, w, h); // Set viewport to FBO dimensions
    gl.clear(gl.COLOR_BUFFER_BIT); // Clear the FBO's color buffer

    let texelSizeX = 1.0 / w;
    let texelSizeY = 1.0 / h;

    return {
        texture,
        fbo,
        width: w,
        height: h,
        texelSizeX,
        texelSizeY,
        /**
         * Attaches this FBO's texture to a specific texture unit.
         * @param {number} id The texture unit ID (e.g., 0, 1, 2).
         * @returns {number} The texture unit ID.
         */
        attach(id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };
}

/**
 * Creates a "double FBO" for ping-ponging (reading from one, writing to another, then swapping).
 * Essential for iterative simulations like fluid dynamics.
 * @param {number} w Width of the textures.
 * @param {number} h Height of the textures.
 * @param {number} internalFormat Internal format of the textures.
 * @param {number} format Format of the textures.
 * @param {number} type Data type of the textures.
 * @param {number} param Texture filtering parameter.
 * @returns {object} Double FBO object with read, write, and swap methods.
 */
function createDoubleFBO(w, h, internalFormat, format, type, param) {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);

    return {
        width: w,
        height: h,
        texelSizeX: fbo1.texelSizeX,
        texelSizeY: fbo1.texelSizeY,
        get read() {
            return fbo1;
        },
        set read(value) {
            fbo1 = value;
        },
        get write() {
            return fbo2;
        },
        set write(value) {
            fbo2 = value;
        },
        swap() {
            let temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;
        }
    }
}

/**
 * Resizes a single FBO by creating a new one and copying content from the old.
 * @param {object} target The FBO to resize.
 * @param {number} w New width.
 * @param {number} h New height.
 * @param {number} internalFormat Internal format.
 * @param {number} format Format.
 * @param {number} type Type.
 * @param {number} param Filtering parameter.
 * @returns {object} The new, resized FBO.
 */
function resizeFBO(target, w, h, internalFormat, format, type, param) {
    let newFBO = createFBO(w, h, internalFormat, format, type, param);
    copyProgram.bind(); // Use copy shader
    gl.uniform1i(copyProgram.uniforms.uTexture, target.attach(0)); // Bind old texture as source
    blit(newFBO); // Copy to new FBO
    return newFBO;
}

/**
 * Resizes a double FBO, resizing its read buffer and creating a new write buffer.
 * @param {object} target The double FBO to resize.
 * @param {number} w New width.
 * @param {number} h New height.
 * @param {number} internalFormat Internal format.
 * @param {number} format Format.
 * @param {number} type Type.
 * @param {number} param Filtering parameter.
 * @returns {object} The resized double FBO.
 */
function resizeDoubleFBO(target, w, h, internalFormat, format, type, param) {
    if (target.width === w && target.height === h) {
        return target; // No resize needed
    }
    // Resize the read buffer and create a new, empty write buffer
    target.read = resizeFBO(target.read, w, h, internalFormat, format, type, param);
    target.write = createFBO(w, h, internalFormat, format, type, param);
    target.width = w;
    target.height = h;
    target.texelSizeX = 1.0 / w;
    target.texelSizeY = 1.0 / h;
    return target;
}

/**
 * Creates a WebGL texture from an image URL asynchronously.
 * @param {string} url The URL of the image.
 * @returns {object} An object representing the texture, initially a 1x1 white texture.
 */
function createTextureAsync(url) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    // Initialize with a 1x1 white pixel while loading
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255]));

    let obj = {
        texture,
        width: 1,
        height: 1,
        attach(id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };

    let image = new Image();
    image.onload = () => {
        obj.width = image.width;
        obj.height = image.height;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
    };
    image.onerror = () => {
        console.error("Failed to load texture from: " + url);
        // Fallback to a plain white texture if loading fails
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255]));
    };
    image.src = url; // Set the image source
    return obj;
}

/**
 * Updates the keywords for the display material based on current config.
 */
function updateKeywords() {
    let displayKeywords = [];
    if (config.SHADING) displayKeywords.push("SHADING");
    displayMaterial.setKeywords(displayKeywords);
}

updateKeywords();      // Initial update of display shader keywords
initFramebuffers();    // Initial setup of all framebuffers

let lastUpdateTime = Date.now();
let colorUpdateTimer = 0.0;

/**
 * The main animation loop.
 * Calculates delta time, handles resizing, updates colors, applies inputs, steps simulation, and renders.
 */
function update() {
    const dt = calcDeltaTime(); // Calculate time elapsed since last frame

    if (resizeCanvas()) {
        initFramebuffers(); // Re-initialize framebuffers if canvas size changed
    }
    updateColors(dt); // Update colors for new splats
    applyInputs();    // Apply user interaction (mouse/touch splats)
    step(dt);         // Advance the fluid simulation by one step
    render(null);     // Render the final fluid simulation to the canvas

    requestAnimationFrame(update); // Request the next frame
}

/**
 * Calculates the delta time (time elapsed since the last frame) in seconds.
 * Caps the delta time to prevent large jumps on tab switching or heavy load.
 * @returns {number} Delta time in seconds.
 */
function calcDeltaTime() {
    let now = Date.now();
    let dt = (now - lastUpdateTime) / 1000;
    dt = Math.min(dt, 0.016666); // Cap dt to ~60fps frame time (1/60th second)
    lastUpdateTime = now;
    return dt;
}

/**
 * Resizes the canvas element to match its client dimensions (CSS size).
 * Returns true if a resize occurred, false otherwise.
 * @returns {boolean} True if resized, false if not.
 */
function resizeCanvas() {
    let width = scaleByPixelRatio(canvas.clientWidth);
    let height = scaleByPixelRatio(canvas.clientHeight);
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        return true;
    }
    return false;
}

/**
 * Updates the colors of the pointers over time.
 * @param {number} dt Delta time.
 */
function updateColors(dt) {
    colorUpdateTimer += dt * config.COLOR_UPDATE_SPEED;
    if (colorUpdateTimer >= 1) {
        colorUpdateTimer = wrap(colorUpdateTimer, 0, 1); // Wrap the timer
        pointers.forEach(p => {
            p.color = generateColor(); // Generate a new color for each pointer
        });
    }
}

/**
 * Applies active user inputs (splats) to the simulation.
 */
function applyInputs() {
    pointers.forEach(p => {
        if (p.moved) {
            p.moved = false; // Reset moved flag
            splatPointer(p); // Apply splat for the pointer
        }
    });
}

/**
 * Performs one step of the fluid simulation.
 * This involves several shader passes to simulate advection, diffusion, pressure, and vorticity.
 * @param {number} dt Delta time.
 */
function step(dt) {
    gl.disable(gl.BLEND); // Blending is typically disabled for simulation passes

    // 1. Calculate Curl (Vorticity)
    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(curl); // Output curl to 'curl' FBO

    // 2. Apply Vorticity Confinement
    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl.attach(1));
    gl.uniform1f(vorticityProgram.uniforms.curl, config.CURL);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write); // Output modified velocity to 'velocity.write'
    velocity.swap();       // Swap velocity FBOs for next step

    // 3. Calculate Divergence
    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence); // Output divergence to 'divergence' FBO

    // 4. Clear Pressure Field
    clearProgram.bind();
    gl.uniform1i(clearProgram.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, config.PRESSURE);
    blit(pressure.write); // Clear pressure to a dissipated value
    pressure.swap();       // Swap pressure FBOs

    // 5. Solve Pressure (Jacobi Iterations)
    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence.attach(0));
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
        gl.uniform1i(pressureProgram.uniforms.uPressure, pressure.read.attach(1));
        blit(pressure.write); // Iterate pressure solution
        pressure.swap();       // Swap pressure FBOs after each iteration
    }

    // 6. Subtract Pressure Gradient from Velocity
    gradienSubtractProgram.bind();
    gl.uniform2f(gradienSubtractProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(gradienSubtractProgram.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(gradienSubtractProgram.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write); // Output divergence-free velocity to 'velocity.write'
    velocity.swap();       // Swap velocity FBOs

    // 7. Advect Velocity (move velocity field with itself)
    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    if (!ext.supportLinearFiltering) {
        gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    }
    let velocityId = velocity.read.attach(0);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocityId);
    gl.uniform1i(advectionProgram.uniforms.uSource, velocityId);
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, config.VELOCITY_DISSIPATION);
    blit(velocity.write); // Advect velocity
    velocity.swap();       // Swap velocity FBOs

    // 8. Advect Dye (move dye with velocity field)
    if (!ext.supportLinearFiltering) {
        gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
    }
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(advectionProgram.uniforms.dissipation, config.DENSITY_DISSIPATION);
    blit(dye.write); // Advect dye
    dye.swap();       // Swap dye FBOs
}

/**
 * Renders the fluid simulation to the target (usually the canvas).
 * Enables blending for displaying transparent fluid over the background.
 * @param {object} target The FBO to render to, or null for the canvas.
 */
function render(target) {
    // Set blending function for additive blending (colors add up)
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.BLEND); // Enable blending

    drawDisplay(target); // Draw the final fluid display
}

/**
 * Draws the fluid density texture to the display target (canvas).
 * @param {object} target The FBO to render to, or null for the canvas.
 */
function drawDisplay(target) {
    let width = target == null ? gl.drawingBufferWidth : target.width;
    let height = target == null ? gl.drawingBufferHeight : target.height;

    displayMaterial.bind(); // Use the display shader
    if (config.SHADING) {
        gl.uniform2f(displayMaterial.uniforms.texelSize, 1.0 / width, 1.0 / height);
    }
    gl.uniform1i(displayMaterial.uniforms.uTexture, dye.read.attach(0)); // Bind dye texture as source
    blit(target); // Draw to target
}

/**
 * Applies a splat (impulse) based on a pointer's movement data.
 * @param {object} pointer The pointer object containing movement and color data.
 */
function splatPointer(pointer) {
    let dx = pointer.deltaX * config.SPLAT_FORCE; // Scale delta by splat force
    let dy = pointer.deltaY * config.SPLAT_FORCE;
    splat(pointer.texcoordX, pointer.texcoordY, dx, dy, pointer.color); // Apply the splat
}

/**
 * Applies a "click" splat with predefined velocity and color.
 * @param {object} pointer The pointer object at the click location.
 */
function clickSplat(pointer) {
    const color = generateColor(); // Generate a new color for the click
    // Amplify color for more visibility on click
    color.r *= 10.0;
    color.g *= 10.0;
    color.b *= 10.0;
    let dx = 10 * (Math.random() - 0.5); // Random initial velocity for clicks
    let dy = 30 * (Math.random() - 0.5);
    splat(pointer.texcoordX, pointer.texcoordY, dx, dy, color); // Apply the splat
}

/**
 * Applies a splat (impulse) of velocity and color at a specific location.
 * @param {number} x X-coordinate of the splat (normalized UV).
 * @param {number} y Y-coordinate of the splat (normalized UV).
 * @param {number} dx X-component of velocity impulse.
 * @param {number} dy Y-component of velocity impulse.
 * @param {object} color RGB color object for the dye splat.
 */
function splat(x, y, dx, dy, color) {
    // Splat velocity
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x, y);
    gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0.0); // Velocity as color components (XY)
    gl.uniform1f(splatProgram.uniforms.radius, correctRadius(config.SPLAT_RADIUS / 100.0));
    blit(velocity.write);
    velocity.swap();

    // Splat dye (color)
    gl.uniform1i(splatProgram.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(splatProgram.uniforms.color, color.r, color.g, color.b); // RGB color for dye
    blit(dye.write);
    dye.swap();
}

/**
 * Corrects the splat radius for canvas aspect ratio.
 * @param {number} radius The base radius.
 * @returns {number} The adjusted radius.
 */
function correctRadius(radius) {
    let aspectRatio = canvas.width / canvas.height;
    if (aspectRatio > 1) {
        radius *= aspectRatio; // Increase radius on wider screens
    }
    return radius;
}

// --- Event Listeners for Mouse and Touch Interaction ---

// Mouse down event: starts a splat
window.addEventListener('mousedown', e => {
    let pointer = pointers[0]; // Assuming only one mouse pointer
    let posX = scaleByPixelRatio(e.clientX);
    let posY = scaleByPixelRatio(e.clientY);
    updatePointerDownData(pointer, -1, posX, posY); // -1 for mouse ID
    clickSplat(pointer); // Apply an initial click splat
});

// Mouse move event: continues splat if mouse is down
window.addEventListener('mousemove', e => {
    let pointer = pointers[0];
    let posX = scaleByPixelRatio(e.clientX);
    let posY = scaleByPixelRatio(e.clientY);
    let color = pointer.color; // Use the current color for continuous movement
    // Only update movement data if the pointer is "down" (mouse button held)
    if (pointer.down) {
        updatePointerMoveData(pointer, posX, posY, color);
    }
});

// Touch start event: starts multiple splats for multiple touches
window.addEventListener('touchstart', e => {
    e.preventDefault(); // Prevent default touch actions (e.g., scrolling, zooming)
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        // Ensure there's a pointer object for each touch
        if (i >= pointers.length) {
            pointers.push(new pointerPrototype());
        }
        let pointer = pointers[i];
        let posX = scaleByPixelRatio(touches[i].clientX);
        let posY = scaleByPixelRatio(touches[i].clientY);
        updatePointerDownData(pointer, touches[i].identifier, posX, posY);
    }
}, { passive: false }); // `passive: false` to allow `e.preventDefault()`

// Touch move event: continues splats for active touches
window.addEventListener('touchmove', e => {
    e.preventDefault(); // Prevent default touch actions
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        // Find the corresponding pointer object for the current touch
        let pointer = pointers.find(p => p.id === touches[i].identifier);
        if (pointer) { // Ensure pointer exists before updating
            let posX = scaleByPixelRatio(touches[i].clientX);
            let posY = scaleByPixelRatio(touches[i].clientY);
            updatePointerMoveData(pointer, posX, posY, pointer.color);
        }
    }
}, { passive: false }); // `passive: false` to allow `e.preventDefault()`

// Touch end event: ends splat for lifted touches
window.addEventListener('touchend', e => {
    const touches = e.changedTouches; // Touches that changed status (lifted)
    for (let i = 0; i < touches.length; i++) {
        // Find the corresponding pointer object for the ended touch
        let pointerIndex = pointers.findIndex(p => p.id === touches[i].identifier);
        if (pointerIndex !== -1) {
            let pointer = pointers[pointerIndex];
            updatePointerUpData(pointer);
            // Optionally, remove the pointer from the array if it's no longer needed after touch up
            // pointers.splice(pointerIndex, 1);
        }
    }
});

// Mouse leave event: ends splat for mouse pointer when it leaves window
window.addEventListener('mouseleave', () => {
    // Assuming pointers[0] is always the mouse pointer
    pointers[0].down = false;
});


/**
 * Updates a pointer's data when it goes down (mouse click or touch start).
 * @param {object} pointer The pointer object to update.
 * @param {number} id The unique ID of the pointer/touch.
 * @param {number} posX X-coordinate in canvas pixels.
 * @param {number} posY Y-coordinate in canvas pixels.
 */
function updatePointerDownData(pointer, id, posX, posY) {
    pointer.id = id;
    pointer.down = true;
    pointer.moved = false;
    pointer.texcoordX = posX / canvas.width;
    pointer.texcoordY = 1.0 - posY / canvas.height; // Invert Y for WebGL coordinates
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.deltaX = 0;
    pointer.deltaY = 0;
    pointer.color = generateColor(); // Generate a new vibrant color on down
}

/**
 * Updates a pointer's data when it moves.
 * @param {object} pointer The pointer object to update.
 * @param {number} posX Current X-coordinate in canvas pixels.
 * @param {number} posY Current Y-coordinate in canvas pixels.
 * @param {object} color The current color for the splat.
 */
function updatePointerMoveData(pointer, posX, posY, color) {
    // Only update if pointer is actually "down" (e.g., mouse button held or touch active)
    if (!pointer.down) return;

    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.texcoordX = posX / canvas.width;
    pointer.texcoordY = 1.0 - posY / canvas.height; // Invert Y for WebGL coordinates
    pointer.deltaX = correctDeltaX(pointer.texcoordX - pointer.prevTexcoordX);
    pointer.deltaY = correctDeltaY(pointer.texcoordY - pointer.prevTexcoordY);
    pointer.moved = Math.abs(pointer.deltaX) > 0 || Math.abs(pointer.deltaY) > 0; // Check if significant movement occurred
    pointer.color = color;
}

/**
 * Updates a pointer's data when it is lifted (mouse up or touch end).
 * @param {object} pointer The pointer object to update.
 */
function updatePointerUpData(pointer) {
    pointer.down = false;
}

/**
 * Corrects the delta X value based on canvas aspect ratio.
 * @param {number} delta The raw delta X.
 * @returns {number} The corrected delta X.
 */
function correctDeltaX(delta) {
    let aspectRatio = canvas.width / canvas.height;
    if (aspectRatio < 1) {
        delta *= aspectRatio;
    }
    return delta;
}

/**
 * Corrects the delta Y value based on canvas aspect ratio.
 * @param {number} delta The raw delta Y.
 * @returns {number} The corrected delta Y.
 */
function correctDeltaY(delta) {
    let aspectRatio = canvas.width / canvas.height;
    if (aspectRatio > 1) {
        delta /= aspectRatio;
    }
    return delta;
}

/**
 * Generates a random vibrant color in RGB format using HSV to RGB conversion.
 * @returns {{r: number, g: number, b: number}} An object with R, G, B color components (0-1 range).
 */
function generateColor() {
    // Generate a random hue (0 to 1), with full saturation (1.0) and value (1.0) for vibrancy
    let c = HSVtoRGB(Math.random(), 1.0, 1.0);
    // Further adjust brightness for the fluid effect, keeping it somewhat dim unless forces are strong
    // This multiplier can be tuned for desired initial brightness.
    c.r *= 0.15;
    c.g *= 0.15;
    c.b *= 0.15;
    return c;
}

/**
 * Converts HSV (Hue, Saturation, Value) color to RGB.
 * @param {number} h Hue (0-1).
 * @param {number} s Saturation (0-1).
 * @param {number} v Value (brightness) (0-1).
 * @returns {{r: number, g: number, b: number}} An object with R, G, B color components (0-1 range).
 */
function HSVtoRGB(h, s, v) {
    let r, g, b, i, f, p, q, t;
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return {
        r,
        g,
        b
    };
}

/**
 * Wraps a value within a min/max range (like modulo).
 * @param {number} value The value to wrap.
 * @param {number} min The minimum boundary.
 * @param {number} max The maximum boundary.
 * @returns {number} The wrapped value.
 */
function wrap(value, min, max) {
    let range = max - min;
    if (range === 0) return min; // Avoid division by zero
    return (value - min) % range + min;
}

/**
 * Calculates the appropriate width and height for a simulation texture
 * based on a given resolution and the canvas aspect ratio.
 * @param {number} resolution The target resolution (e.g., SIM_RESOLUTION, DYE_RESOLUTION).
 * @returns {{width: number, height: number}} Object with calculated width and height.
 */
function getResolution(resolution) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1) {
        aspectRatio = 1.0 / aspectRatio; // Ensure aspect ratio is always >= 1
    }

    let min = Math.round(resolution);
    let max = Math.round(resolution * aspectRatio);

    if (gl.drawingBufferWidth > gl.drawingBufferHeight) {
        return { width: max, height: min };
    } else {
        return { width: min, height: max };
    }
}

/**
 * Scales an input value by the device's pixel ratio.
 * Useful for handling high-DPI displays.
 * @param {number} input The value to scale.
 * @returns {number} The scaled value.
 */
function scaleByPixelRatio(input) {
    let pixelRatio = window.devicePixelRatio || 1;
    return Math.floor(input * pixelRatio);
}

/**
 * Calculates a simple hash code for a string.
 * Used for caching shader programs based on keywords.
 * @param {string} s The input string.
 * @returns {number} The hash code.
 */
function hashCode(s) {
    if (s.length === 0) return 0;
    let hash = 0;
    for (let i = 0; i < s.length; i++) {
        hash = (hash << 5) - hash + s.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
}

// Start the animation loop once the window content is fully loaded
window.onload = function() {
    update();
};
