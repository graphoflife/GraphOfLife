/*
 * The front page's screen: an old tube television, which shows whichever
 * backdrop is on — the eyes, or the recorded world — the way such a tube
 * would. The words lie over it and stay crisp, and so does the world's wall
 * of digits: no text is seen through the glass.
 *
 * It is made the way the emulators of old screens make theirs (crt-lottes,
 * crt-royale, crt-geom): the picture is drawn as usual into a canvas of its
 * own, kept out of sight, and one pass on the graphics card shows it through
 * the tube, every pixel of the window at once:
 *
 *  - the glass is a rounded square — |X|⁶ + |Y|⁶ = 1 on the stage's
 *    half-sizes — that bulges, so the picture bends in towards the corners
 *    (the eyes are drawn bent already, through warp(); the world is bent
 *    here);
 *  - the picture is laid down in scan lines, bent with the bulge, one along
 *    every row of the eyes' raster, so that each row of pixel-eyes shines on
 *    a line of its own; a dark line is a thin one, and a bright line spreads
 *    into the dark between;
 *  - the light comes through a grille of deep red, green and blue phosphor
 *    stripes, three of them about a line wide, so that every line is a row
 *    of little red, green and blue pixels; the grille is as deep all over
 *    the glass, as on a real tube;
 *  - the three beams miss each other a little, the red and the blue
 *    outwards, and lose their focus, the more the further from the middle;
 *  - bright light spreads in the thick glass round it (halation);
 *  - the lit glass glows faintly in the phosphor's colour, the corners fall
 *    darker, a still grain lies on the phosphor, and on a dark screen a sheen
 *    sits high on the left, where a window would catch it.
 *
 * All of it covers the whole screen, and the rest grows stronger out to the
 * glass's edge, and at its full strength from there into the window's corners, so
 * there is no frame and no empty corner. On Paper the dark between the lines
 * and the stripes is the eyes' blue rather than black.
 *
 * Nothing on it moves of itself: no flicker and no rolling band. It is drawn
 * again only when the picture under it has been (mark()), at most as often as
 * that picture is. Where the browser has no WebGL 2 the pictures show as
 * they are, without the tube.
 */
const Tube = {
  /** How far the picture bends in towards the corners of the bulging glass, as a share of the half-width. */
  BULGE: 0.12,

  /** How strongly the tube shows in the middle of the glass, against its full strength at the edge. */
  MIDDLE: 0.72,

  /** How strongly it shows over the great eye, against everywhere else; and how many of the great eye's radii out it is back to its strength there, smoothly. */
  QUIET: 0.3,
  QUIET_TO: 12,

  /** How much light is left halfway between two scan lines, at full strength: where the picture is dark, and where it is bright. */
  GAP_DARK: 0.05,
  GAP_BRIGHT: 0.3,

  /** How deep the phosphor stripes shade the light: a colour keeps 1 − MASK of itself on the other two colours' stripes. */
  MASK: 0.75,

  /** How much brighter a colour is on its own stripe. */
  MASK_LIGHT: 1.45,

  /** The light the lines and the stripes take away, given back; what would be brighter than white is eased into it. */
  BOOST: 1.35,

  /** How far apart the red and the blue beams land from the green, in pixels: everywhere, and on top of that out at the glass's edge. */
  SPLIT: 0.6,
  FRINGE: 1.6,

  /** How soft the picture is, in pixels: in the middle, and on top of that out at the glass's edge. */
  SOFT: 0.4,
  BLUR: 0.9,

  /** How much of the picture's light spreads into the glass round it. */
  HALATION: 0.55,

  /** The lit glass's glow, in the phosphor's colour: in the middle, and at the edge. */
  WASH: 0.05,
  WASH_EDGE: 0.14,

  /** How much darker the window's very corners are. */
  CORNERS: 0.4,

  /** The phosphor's still grain. */
  GRAIN: 0.07,

  /** The sheen of a window on a dark glass. */
  SHEEN: 0.07,

  canvas: null,
  host: null,
  width: 0,
  height: 0,
  ax: 0,
  ay: 0,
  pitch: 6,

  gl: null,
  picture: null,
  bent: true,
  marked: false,

  /** Where the great eye shows on the glass and its radius, in pixels of the page; no radius while the eyes are not shown. */
  quietX: 0,
  quietY: 0,
  quietR: 0,

  init(canvas, host) {
    this.canvas = canvas;
    this.host = host;
    if (!canvas) return;
    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      this.gl = null;
      this.host?.classList.remove('tube-on');
    });
    canvas.addEventListener('webglcontextrestored', () => this.start());
    this.start();
  },

  /**
   * The glass for a screen of this size, and the pitch both its scan lines
   * and the eyes' raster keep to: a little coarser on a bigger screen, and a
   * whole number of the canvas's own pixels, six at least, so that the lines
   * lie evenly, none of them thicker than the next, with dark between them.
   */
  shape(width, height) {
    this.width = width;
    this.height = height;
    this.ax = width / 2;
    this.ay = height / 2;
    const scale = Math.min(2, window.devicePixelRatio || 1);
    const want = Math.max(4, Math.min(7, Math.min(width, height) / 170));
    this.pitch = Math.max(6, Math.round(want * scale)) / scale;
  },

  /**
   * How far a point of the picture lies inside the glass, near enough: how far
   * short of the outline it is, over how fast that changes. Negative outside
   * it. Leaves the way inwards in `inX`, `inY`.
   */
  face(x, y) {
    const X = (x - this.width / 2) / this.ax, Y = (y - this.height / 2) / this.ay;
    const X2 = X * X, Y2 = Y * Y, X4 = X2 * X2, Y4 = Y2 * Y2, sum = X4 * X2 + Y4 * Y2;
    if (sum < 1e-12) { this.inX = 0; this.inY = 0; return Math.min(this.ax, this.ay); }
    const rho = Math.cbrt(Math.sqrt(sum)), r5 = rho * rho * rho * rho * rho;
    const gx = X4 * X / (this.ax * r5), gy = Y4 * Y / (this.ay * r5);
    const slope = Math.sqrt(gx * gx + gy * gy) || 1e-9;
    this.inX = -gx / slope;
    this.inY = -gy / slope;
    return (1 - rho) / slope;
  },

  /**
   * Where a point of the picture shows on the bulging glass, left in `atX`,
   * `atY`: pulled in towards the corners, the more the further out. The
   * shader below bends the same way.
   */
  warp(x, y) {
    const hw = this.width / 2, hh = this.height / 2, b = this.BULGE;
    const X = (x - hw) / hw, Y = (y - hh) / hh;
    this.atX = hw + X * (1 - b * Y * Y) * hw;
    this.atY = hh + Y * (1 - b * X * X) * hh;
  },

  /** The point of the picture that shows at (x, y) on the glass, left in `atX`, `atY`: the bulge undone, by a few steps of guessing. */
  unwarp(x, y) {
    const hw = this.width / 2, hh = this.height / 2, b = this.BULGE;
    const Xs = (x - hw) / hw, Ys = (y - hh) / hh;
    let X = Xs, Y = Ys;
    for (let i = 0; i < 3; i++) {
      X = Xs / (1 - b * Y * Y);
      Y = Ys / (1 - b * X * X);
    }
    this.atX = hw + X * hw;
    this.atY = hh + Y * hh;
  },

  /** A colour written as #rgb or #rrggbb, as red, green and blue. */
  rgb(colour) {
    const hex = String(colour).trim().replace(/^#/, '');
    const full = hex.length === 3 ? [...hex].map(c => c + c).join('') : hex;
    const [r, g, b] = [0, 2, 4].map(i => parseInt(full.slice(i, i + 2), 16));
    return [r, g, b].map(v => (Number.isFinite(v) ? v : 128));
  },

  /** What the tube shows: a canvas, and whether its picture is drawn bent already (the eyes' is). */
  show(picture, bent) {
    this.picture = picture;
    this.bent = bent;
    this.mark();
  },

  /** The great eye stands here, this big (eyes.js): the tube is quieter round it. A radius of 0 for none. */
  quiet(x, y, r) {
    this.quietX = x;
    this.quietY = y;
    this.quietR = r;
    this.mark();
  },

  /** The picture under the glass has been drawn again: so is the tube, on the next frame. */
  mark() {
    this.marked = true;
  },

  /** One animation frame, from Home: drawn if anything under it changed. */
  tick() {
    if (this.marked) this.draw();
  },

  /** Laid out for the stage as it now is: on arrival, and whenever it changes size. */
  resize() {
    const canvas = this.canvas, host = this.host;
    if (!canvas || !host) return;
    const width = host.clientWidth, height = host.clientHeight;
    if (!(width > 0 && height > 0)) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.shape(width, height);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    this.mark();
  },

  /** The theme changed: the same glass in its new inks. */
  repaint() {
    this.mark();
  },

  /**
   * The graphics card made ready: the pass that draws the tube, a triangle
   * that covers the window, and room for the picture. Leaves the stage
   * marked `tube-on`, which hides the pictures' own canvases, only once it
   * all works.
   */
  start() {
    const canvas = this.canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false
    });
    if (!gl) return;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    try {
      const program = gl.createProgram();
      gl.attachShader(program, compile(gl.VERTEX_SHADER, this.VERTEX));
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, this.fragment()));
      gl.bindAttribLocation(program, 0, 'corner');
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      this.program = program;
    } catch (err) {
      console.warn('the tube is not shown:', err.message);
      return;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    this.at = {};
    for (const name of ['uSize', 'uRes', 'uScale', 'uSource', 'uBulge', 'uBent', 'uPitch',
                        'uQuiet', 'uStage', 'uGlow', 'uShadow', 'uSheen']) {
      this.at[name] = gl.getUniformLocation(this.program, name);
    }
    this.gl = gl;
    this.host?.classList.add('tube-on');
    this.mark();
  },

  draw() {
    this.marked = false;
    const gl = this.gl, picture = this.picture;
    if (!gl || !this.width || !this.canvas.width || !picture || !(picture.width > 0 && picture.height > 0)) return;
    // The picture onto the card, with the smaller copies of it that the blur
    // and the halation are read from.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, picture);
    gl.generateMipmap(gl.TEXTURE_2D);

    // The inks: the empty screen, the phosphor's colour, the light a window
    // casts on the glass — and what the dark between the lines and the
    // stripes falls to: black on a dark screen, on Paper the eyes' blue.
    const [stage, glow, ball] = ['stage', 'eyeIris', 'eyeBall'].map(role => this.rgb(Ink.of(role)).map(v => v / 255));
    const light = stage[0] * 0.299 + stage[1] * 0.587 + stage[2] * 0.114 > 0.5;
    const shadow = light ? glow.map(v => v * 0.55) : [0, 0, 0];

    const at = this.at;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.program);
    gl.uniform2f(at.uSize, this.width, this.height);
    gl.uniform2f(at.uRes, this.canvas.width, this.canvas.height);
    gl.uniform1f(at.uScale, this.canvas.width / this.width);
    gl.uniform1f(at.uSource, picture.width / this.width);
    gl.uniform1f(at.uBulge, this.BULGE);
    gl.uniform1f(at.uBent, this.bent ? 1 : 0);
    gl.uniform1f(at.uPitch, this.pitch);
    gl.uniform3f(at.uQuiet, this.quietX, this.quietY, this.quietR);
    gl.uniform3fv(at.uStage, stage);
    gl.uniform3fv(at.uGlow, glow);
    gl.uniform3fv(at.uShadow, shadow);
    gl.uniform3fv(at.uSheen, light ? [0, 0, 0] : ball);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  },

  VERTEX: `#version 300 es
in vec2 corner;
void main() { gl_Position = vec4(corner, 0.0, 1.0); }
`,

  /** The pass itself, with this file's strengths written into it. */
  fragment() {
    const f = v => Number(v).toFixed(4);
    return `#version 300 es
precision highp float;

uniform sampler2D uPicture;  // the backdrop
uniform vec2 uSize;          // the stage, in pixels of the page
uniform vec2 uRes;           // this canvas, in its own pixels
uniform float uScale;        // its pixels to a pixel of the page
uniform float uSource;       // the same for the picture
uniform float uBulge, uBent, uPitch;
uniform vec3 uQuiet;         // where the great eye shows and its radius, in pixels of the page
uniform vec3 uStage, uGlow, uShadow, uSheen;
out vec4 colour;

// The point of the picture that shows at a point of the bulging glass, as
// unwarp() above finds it. The glass runs from -1 to 1 across the stage,
// downwards.
vec2 unwarp(vec2 S) {
  vec2 X = S;
  for (int i = 0; i < 3; i++) {
    X.x = S.x / (1.0 - uBulge * X.y * X.y);
    X.y = S.y / (1.0 - uBulge * X.x * X.x);
  }
  return X;
}

// How far out on the rounded square: 0 in the middle, 1 on its edge.
float outward(vec2 X) {
  vec2 c = abs(X * X * X);
  return pow(c.x * c.x + c.y * c.y, 1.0 / 6.0);
}

// The picture at a point of the glass, softened by \`blur\` pixels, over the
// empty screen. The eyes are drawn bent already; the world is bent here, and
// beyond its edge is the empty screen.
vec3 picture(vec2 X, float blur) {
  vec2 uv = (uBent > 0.5 ? X : unwarp(X)) * 0.5 + 0.5;
  if (uBent < 0.5 && (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))) return uStage;
  vec4 a = textureLod(uPicture, uv, log2(max(1.0, blur * uSource)));
  return a.rgb + uStage * (1.0 - a.a);
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y) / uScale;
  vec2 X = (p - 0.5 * uSize) / (0.5 * uSize);
  vec2 Q = unwarp(X);
  float edge = smoothstep(0.1, 1.0, outward(Q));
  float s = mix(${f(this.MIDDLE)}, 1.0, edge);
  float far = min(dot(X, X), 2.0);

  // Quieter round the great eye: the lines, the stripes, the beams' miss,
  // the softness and the grain come up smoothly from QUIET of their strength
  // over the eye to all of it QUIET_TO radii out.
  float t = uQuiet.z > 0.0
    ? mix(${f(this.QUIET)}, 1.0, smoothstep(1.0, ${f(this.QUIET_TO)}, length(p - uQuiet.xy) / uQuiet.z))
    : 1.0;

  // The three beams: the red and the blue land beside the green, outwards,
  // and all of them lose their focus away from the middle.
  vec2 miss = (vec2(${f(this.SPLIT)} * t, 0.0) + X * ${f(this.FRINGE)} * far) / (0.5 * uSize);
  float blur = ${f(this.SOFT)} * t + ${f(this.BLUR)} * far;
  vec3 c = vec3(picture(X + miss, blur).r, picture(X, blur).g, picture(X - miss, blur).b);

  // The lit glass's own faint glow.
  c = mix(c, uGlow, mix(${f(this.WASH)}, ${f(this.WASH_EDGE)}, edge));

  // The scan lines, along the picture's rows and bent with the glass, a line's
  // middle on each row of the raster (and on a row of the canvas's pixels):
  // the nearest line's middle is d lines away. A dark line is thin; a bright
  // one spreads into the dark between.
  float y = ((Q.y * 0.5 + 0.5) * uSize.y - 0.5 / uScale) / uPitch;
  float d = fract(y) - 0.5;
  vec3 gap = pow(mix(vec3(${f(this.GAP_DARK)}), vec3(${f(this.GAP_BRIGHT)}), clamp(c, 0.0, 1.0)), vec3(s * t));
  vec3 beam = exp2(log2(gap) * 4.0 * d * d);

  // The phosphor's stripes, red, green and blue side by side, a whole number
  // of the screen's pixels each and three of them about a line wide, so that
  // every line is a row of little red, green and blue pixels. They are as
  // deep all over the glass, as on a real tube.
  float stripe = max(1.0, floor(uPitch * uScale / 3.0 + 0.5));
  float k = mod(floor(gl_FragCoord.x / stripe), 3.0);
  vec3 own = vec3(equal(vec3(k), vec3(0.0, 1.0, 2.0)));
  vec3 grille = mix(vec3(1.0), mix(vec3(1.0 - ${f(this.MASK)}), vec3(${f(this.MASK_LIGHT)}), own), t);

  // The light they take away given back, as much as they take.
  vec3 lit = uShadow + (c - uShadow) * beam * grille * mix(1.0, ${f(this.BOOST)}, t);

  // Light spread round whatever is brighter than the empty screen: near it,
  // the beam's own spread, which lights the phosphor's stripes too; further
  // out, the glow in the thick glass.
  vec3 halo = max(picture(X, 7.0) - uStage, 0.0) * 0.55 * grille + max(picture(X, 22.0) - uStage, 0.0) * 0.45;
  lit += halo * ${f(this.HALATION)} * s;

  // Darker into the corners.
  lit = uShadow + (lit - uShadow) * (1.0 - ${f(this.CORNERS)} * smoothstep(0.35, 1.3, outward(X)));

  // A window's sheen, high on the left.
  vec2 h = (X - vec2(-0.45, -0.62)) * vec2(1.0, 1.5);
  lit += uSheen * ${f(this.SHEEN)} * exp(-4.0 * dot(h, h));

  // The phosphor's grain, the same every frame.
  uvec2 q = uvec2(gl_FragCoord.xy);
  uint n = q.x * 1973u + q.y * 9277u;
  n = (n ^ (n >> 13u)) * 1103515245u;
  n ^= n >> 16u;
  lit += (float(n & 1023u) / 1023.0 - 0.5) * ${f(this.GRAIN)} * s * t;

  // Eased into white, not cut off at it, so a bright eye keeps its lines.
  vec3 over = max(lit - 0.75, 0.0);
  lit = min(lit, 0.75) + 0.25 * (1.0 - exp(-over / 0.25));
  colour = vec4(clamp(lit, 0.0, 1.0), 1.0);
}
`;
  }
};
