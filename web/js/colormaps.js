/*
 * Colour maps.
 *
 * Each map is a list of RGB anchors sampled evenly over [0, 1]; lookup does a
 * linear interpolation between the two nearest anchors. Anchor sets are taken
 * from the matplotlib maps so the web viewer and any PNG output agree.
 */
const COLORMAPS = {
  viridis: [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]],
  plasma:  [[13,8,135],[75,3,161],[125,3,168],[168,34,150],[203,70,121],[229,107,93],[248,148,65],[253,195,40],[240,249,33]],
  magma:   [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]],
  inferno: [[0,0,4],[31,12,72],[85,15,109],[136,34,106],[186,54,85],[227,89,51],[249,142,9],[249,201,50],[252,255,164]],
  cividis: [[0,32,76],[0,60,109],[47,87,118],[86,110,116],[124,135,109],[165,162,98],[209,191,79],[254,223,55]],
  turbo:   [[48,18,59],[70,107,227],[41,177,220],[75,226,146],[166,249,71],[229,211,44],[250,142,42],[220,60,15],[122,4,3]],
  grayscale: [[0,0,0],[255,255,255]],
  coolwarm: [[59,76,192],[110,131,220],[168,180,236],[221,221,221],[238,181,155],[229,131,102],[180,4,38]],
  spectral: [[158,1,66],[213,62,79],[244,109,67],[253,174,97],[254,224,139],[230,245,152],[171,221,164],[102,194,165],[50,136,189],[94,79,162]],
  ember:   [[10,10,20],[60,20,40],[130,30,40],[200,70,30],[245,150,40],[255,230,140]]
};

const COLORMAP_NAMES = Object.keys(COLORMAPS);

/** Sample a colour map at t in [0,1]. Returns [r,g,b]. */
function sampleColormap(name, t, reverse) {
  const anchors = COLORMAPS[name] || COLORMAPS.viridis;
  let x = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  if (reverse) x = 1 - x;

  const scaled = x * (anchors.length - 1);
  const i = Math.min(anchors.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = anchors[i], b = anchors[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f)
  ];
}

function colormapCss(name, t, alpha, reverse) {
  const [r, g, b] = sampleColormap(name, t, reverse);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Fill a <select> with the available colour map names. */
function populateColormapSelect(select, selected) {
  select.innerHTML = '';
  for (const name of COLORMAP_NAMES) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === selected) opt.selected = true;
    select.appendChild(opt);
  }
}

/** Draw a horizontal colour ramp, used for the legend. */
function drawColormapStrip(ctx, x, y, w, h, name, reverse) {
  for (let i = 0; i < w; i++) {
    const [r, g, b] = sampleColormap(name, i / (w - 1), reverse);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x + i, y, 1, h);
  }
}
