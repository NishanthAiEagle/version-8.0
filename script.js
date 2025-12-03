/* script.js — Full file (camera-init + earring placement + robust gallery close)
   Uses numbered files:
   gold_earrings/1.png … N.png
   gold_necklaces/1.png … N.png
   diamond_earrings/1.png … N.png
   diamond_necklaces/1.png … N.png
*/

/* ====== CONFIG: how many images per folder ====== */
const IMAGE_COUNTS = {
  gold_earrings: 5,
  gold_necklaces: 5,
  diamond_earrings: 5,
  diamond_necklaces: 5
};

/* DOM refs */
const videoElement   = document.getElementById('webcam');
const canvasElement  = document.getElementById('overlay');
const canvasCtx      = canvasElement.getContext('2d');

const tryAllBtn      = document.getElementById('tryall-btn');
const flashOverlay   = document.getElementById('flash-overlay');
const galleryModal   = document.getElementById('gallery-modal');
const galleryMain    = document.getElementById('gallery-main');
const galleryThumbs  = document.getElementById('gallery-thumbs');
const galleryClose   = document.getElementById('gallery-close');

let earSizeRange   = document.getElementById('earSizeRange');
let earSizeVal     = document.getElementById('earSizeVal');
let neckYRange     = document.getElementById('neckYRange');
let neckYVal       = document.getElementById('neckYVal');
let neckScaleRange = document.getElementById('neckScaleRange');
let neckScaleVal   = document.getElementById('neckScaleVal');
let posSmoothRange = document.getElementById('posSmoothRange');
let posSmoothVal   = document.getElementById('posSmoothVal');
let earSmoothRange = document.getElementById('earSmoothRange');
let earSmoothVal   = document.getElementById('earSmoothVal');
let debugToggle    = document.getElementById('debugToggle');

/* If tuning panel elements removed from DOM, create safe fallbacks to avoid runtime errors */
if (!earSizeRange) {
  earSizeRange = document.createElement('input'); earSizeRange.value = '0.24';
  earSizeVal = { textContent: '0.24' };
  neckYRange = document.createElement('input'); neckYRange.value = '0.95';
  neckYVal = { textContent: '0.95' };
  neckScaleRange = document.createElement('input'); neckScaleRange.value = '.98';
  neckScaleVal = { textContent: '.98' };
  posSmoothRange = document.createElement('input'); posSmoothRange.value = '0.88';
  posSmoothVal = { textContent: '0.88' };
  earSmoothRange = document.createElement('input'); earSmoothRange.value = '0.90';
  earSmoothVal = { textContent: '0.90' };
  debugToggle = document.createElement('div');
}

/* State & assets */
let earringImg = null, necklaceImg = null;
let currentType = '';
let smoothedLandmarks = null;
let lastPersonSegmentation = null;
let bodyPixNet = null;
let lastBodyPixRun = 0;
let lastSnapshotDataURL = '';

/* Tunables (initial values match UI defaults) */
let EAR_SIZE_FACTOR = parseFloat(earSizeRange.value || 0.24);
let NECK_Y_OFFSET_FACTOR = parseFloat(neckYRange.value || 0.95);
let NECK_SCALE_MULTIPLIER = parseFloat(neckScaleRange.value || 1.15);
let POS_SMOOTH = parseFloat(posSmoothRange.value || 0.88);
let EAR_DIST_SMOOTH = parseFloat(earSmoothRange.value || 0.90);

/* smoothing state */
const smoothedState = { leftEar: null, rightEar: null, neckPoint: null, angle: 0, earDist: null, faceShape: 'unknown' };
const angleBuffer = [];
const ANGLE_BUFFER_LEN = 5;

/* BodyPix loading flag */
let bodyPixNetLoaded = false;

/* watermark image (used in snapshots and overlays) */
const watermarkImg = new Image();
watermarkImg.src = "logo_watermark.png";
watermarkImg.crossOrigin = "anonymous";

/* Utility helpers */
function loadImage(src) {
  return new Promise(res => {
    const i = new Image();
    i.crossOrigin = 'anonymous';
    i.src = src;
    i.onload = () => res(i);
    i.onerror = () => res(null);
  });
}
function toPxX(normX) { return normX * canvasElement.width; }
function toPxY(normY) { return normY * canvasElement.height; }
function lerp(a,b,t) { return a*t + b*(1-t); }
function lerpPt(a,b,t) { return { x: lerp(a.x,b.x,t), y: lerp(a.y,b.y,t) }; }

/* Load BodyPix (non-blocking) */
async function ensureBodyPixLoaded() {
  if (bodyPixNetLoaded) return;
  try {
    bodyPixNet = await bodyPix.load({ architecture:'MobileNetV1', outputStride:16, multiplier:0.5, quantBytes:2 });
    bodyPixNetLoaded = true;
  } catch(e) {
    console.warn('BodyPix load failed', e);
    bodyPixNetLoaded = false;
  }
}

/* Throttled segmentation */
async function runBodyPixIfNeeded(){
  const throttle = 300; // ms
  const now = performance.now();
  if (!bodyPixNetLoaded) return;
  if (now - lastBodyPixRun < throttle) return;
  lastBodyPixRun = now;
  try {
    const seg = await bodyPixNet.segmentPerson(videoElement, { internalResolution:'low', segmentationThreshold:0.7 });
    lastPersonSegmentation = { data: seg.data, width: seg.width, height: seg.height };
  } catch(e) {
    console.warn('BodyPix segmentation error', e);
  }
}

/* ---------- FACE MESH SETUP ---------- */
const faceMesh = new FaceMesh({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
faceMesh.setOptions({ maxNumFaces:1, refineLandmarks:true, minDetectionConfidence:0.6, minTrackingConfidence:0.6 });
faceMesh.onResults(onFaceMeshResults);

/* Start process: request camera permission explicitly, then create Camera helper */
async function initCameraAndModels() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: 1280, height: 720 },
      audio: false
    });
    videoElement.srcObject = stream;
    videoElement.muted = true;
    videoElement.playsInline = true;

    videoElement.style.display = 'block';
    videoElement.style.position = 'fixed';
    videoElement.style.top = '12px';
    videoElement.style.left = '12px';
    videoElement.style.width = '320px';
    videoElement.style.zIndex = '99999';

    await videoElement.play();

    const cameraHelper = new Camera(videoElement, {
      onFrame: async () => { await faceMesh.send({ image: videoElement }); },
      width: 1280,
      height: 720
    });
    cameraHelper.start();

    videoElement.style.display = 'none';

    ensureBodyPixLoaded();

    console.log('✅ Camera stream started and FaceMesh helper initialized.');
  } catch (err) {
    console.error('Camera init error:', err);
    if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
      alert('Please allow camera access for this site (click the camera icon in your browser URL bar).');
    } else if (err.name === 'NotFoundError') {
      alert('No camera found. Please connect a camera and try again.');
    } else {
      alert('Camera initialization failed: ' + (err && err.message ? err.message : err));
    }
  }
}
initCameraAndModels();

/* FaceMesh results handler */
async function onFaceMeshResults(results) {
  if (videoElement.videoWidth && videoElement.videoHeight) {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  }

  canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);
  try { canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height); } catch(e) {}

  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks.length) {
    smoothedLandmarks = null;
    drawWatermark(canvasCtx);
    return;
  }

  const landmarks = results.multiFaceLandmarks[0];

  if (!smoothedLandmarks) smoothedLandmarks = landmarks;
  else {
    smoothedLandmarks = smoothedLandmarks.map((prev,i) => ({
      x: prev.x * 0.72 + landmarks[i].x * 0.28,
      y: prev.y * 0.72 + landmarks[i].y * 0.28,
      z: prev.z * 0.72 + landmarks[i].z * 0.28
    }));
  }

  const leftEar  = { x: toPxX(smoothedLandmarks[132].x), y: toPxY(smoothedLandmarks[132].y) };
  const rightEar = { x: toPxX(smoothedLandmarks[361].x), y: toPxY(smoothedLandmarks[361].y) };
  const neckP    = { x: toPxX(smoothedLandmarks[152].x), y: toPxY(smoothedLandmarks[152].y) };

  let minX=1,minY=1,maxX=0,maxY=0;
  for (let i=0;i<smoothedLandmarks.length;i++){
    const lm = smoothedLandmarks[i];
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }
  const faceWidth = (maxX - minX) * canvasElement.width;
  const faceHeight = (maxY - minY) * canvasElement.height;
  const aspect = faceHeight / (faceWidth || 1);

  let faceShape = 'oval';
  if (aspect < 1.05) faceShape = 'round';
  else if (aspect > 1.25) faceShape = 'long';
  else faceShape = 'oval';
  smoothedState.faceShape = faceShape;

  const rawEarDist = Math.hypot(rightEar.x - leftEar.x, rightEar.y - leftEar.y);
  if (smoothedState.earDist == null) smoothedState.earDist = rawEarDist;
  else smoothedState.earDist = smoothedState.earDist * EAR_DIST_SMOOTH + rawEarDist * (1 - EAR_DIST_SMOOTH);

  if (!smoothedState.leftEar) {
    smoothedState.leftEar = leftEar;
    smoothedState.rightEar = rightEar;
    smoothedState.neckPoint = neckP;
    smoothedState.angle = Math.atan2(rightEar.y - leftEar.y, rightEar.x - leftEar.x);
  } else {
    smoothedState.leftEar  = lerpPt(smoothedState.leftEar,  leftEar,  POS_SMOOTH);
    smoothedState.rightEar = lerpPt(smoothedState.rightEar, rightEar, POS_SMOOTH);
    smoothedState.neckPoint= lerpPt(smoothedState.neckPoint,neckP,   POS_SMOOTH);

    const rawAngle = Math.atan2(rightEar.y - leftEar.y, rightEar.x - leftEar.x);
    let prev = smoothedState.angle;
    let diff = rawAngle - prev;
    if (diff > Math.PI) diff -= 2*Math.PI;
    if (diff < -Math.PI) diff += 2*Math.PI;
    smoothedState.angle = prev + diff * (1 - 0.82);
  }

  angleBuffer.push(smoothedState.angle);
  if (angleBuffer.length > ANGLE_BUFFER_LEN) angleBuffer.shift();
  if (angleBuffer.length > 2) {
    const s = angleBuffer.slice().sort((a,b)=>a-b);
    smoothedState.angle = s[Math.floor(s.length/2)];
  }

  drawJewelrySmart(smoothedState, canvasCtx, smoothedLandmarks, { faceWidth, faceHeight, faceShape });

  await ensureBodyPixLoaded();
  runBodyPixIfNeeded();
  if (lastPersonSegmentation && lastPersonSegmentation.data) {
    compositeHeadOcclusion(canvasCtx, smoothedLandmarks, lastPersonSegmentation);
  } else {
    drawWatermark(canvasCtx);
  }

  if (debugToggle.classList && debugToggle.classList.contains('on')) drawDebugMarkers();
}

/* Core: draw jewelry with shape-aware offsets */
function drawJewelrySmart(state, ctx, landmarks, meta) {
  const leftEar = state.leftEar, rightEar = state.rightEar, neckPoint = state.neckPoint;
  const earDist = state.earDist || Math.hypot(rightEar.x - leftEar.x, rightEar.y - leftEar.y);
  const angle = state.angle || 0;
  const faceShape = meta.faceShape;
  const faceW = meta.faceWidth, faceH = meta.faceHeight;

  let xAdjPx = 0, yAdjPx = 0, sizeMult = 1.0;
  if (faceShape === 'round') {
    xAdjPx = Math.round(faceW * 0.06);
    yAdjPx = Math.round(faceH * 0.02);
    sizeMult = 1.10;
  } else if (faceShape === 'oval') {
    xAdjPx = Math.round(faceW * 0.045);
    yAdjPx = Math.round(faceH * 0.015);
    sizeMult = 1.00;
  } else {
    xAdjPx = Math.round(faceW * 0.04);
    yAdjPx = Math.round(faceH * 0.005);
    sizeMult = 0.95;
  }

  const finalEarringFactor = EAR_SIZE_FACTOR * sizeMult;

  if (earringImg && landmarks) {
    const eWidth = earDist * finalEarringFactor;
    const eHeight = (earringImg.height / earringImg.width) * eWidth;

    const leftCenterX = leftEar.x - xAdjPx;
    const rightCenterX = rightEar.x + xAdjPx;
    const leftCenterY = leftEar.y + (eHeight * 0.18) + yAdjPx;
    const rightCenterY = rightEar.y + (eHeight * 0.18) + yAdjPx;

    const tiltCorrection = - (angle * 0.08);

    ctx.save();
    ctx.translate(leftCenterX, leftCenterY);
    ctx.rotate(tiltCorrection);
    ctx.drawImage(earringImg, -eWidth/2, -eHeight/2, eWidth, eHeight);
    ctx.restore();

    ctx.save();
    ctx.translate(rightCenterX, rightCenterY);
    ctx.rotate(-tiltCorrection);
    ctx.drawImage(earringImg, -eWidth/2, -eHeight/2, eWidth, eHeight);
    ctx.restore();
  }

  if (necklaceImg && landmarks) {
    const nw = earDist * NECK_SCALE_MULTIPLIER;
    const nh = (necklaceImg.height / necklaceImg.width) * nw;
    const yOffset = earDist * NECK_Y_OFFSET_FACTOR;
    ctx.save();
    ctx.translate(neckPoint.x, neckPoint.y + yOffset);
    ctx.rotate(angle);
    ctx.drawImage(necklaceImg, -nw/2, -nh/2, nw, nh);
    ctx.restore();
  }

  drawWatermark(ctx);
}

/* watermark drawing */
function drawWatermark(ctx) {
  try {
    if (watermarkImg && watermarkImg.naturalWidth) {
      const cw = ctx.canvas.width, ch = ctx.canvas.height;
      const w = Math.round(cw * 0.22);
      const h = Math.round((watermarkImg.height / watermarkImg.width) * w);
      ctx.globalAlpha = 0.85;
      ctx.drawImage(watermarkImg, cw - w - 14, ch - h - 14, w, h);
      ctx.globalAlpha = 1;
    }
  } catch(e) {}
}

/* Composite occlusion using BodyPix segmentation */
function compositeHeadOcclusion(mainCtx, landmarks, seg) {
  try {
    const segData = seg.data, segW = seg.width, segH = seg.height;
    const indices = [10,151,9,197,195,4];
    let minX=1,minY=1,maxX=0,maxY=0;
    indices.forEach(i => { 
      const x=landmarks[i].x, y=landmarks[i].y; 
      if (x<minX) minX=x; 
      if(y<minY) minY=y; 
      if(x>maxX) maxX=x; 
      if(y>maxY) maxY=y; 
    });
    const padX = 0.18*(maxX-minX), padY = 0.40*(maxY-minY);
    const L = Math.max(0, (minX - padX) * canvasElement.width);
    const T = Math.max(0, (minY - padY) * canvasElement.height);
    const R = Math.min(canvasElement.width, (maxX + padX) * canvasElement.width);
    const B = Math.min(canvasElement.height, (maxY + padY) * canvasElement.height);
    const W = Math.max(0, R-L), H = Math.max(0, B-T);
    if (W <= 0 || H <= 0) { drawWatermark(mainCtx); return; }

    const off = document.createElement('canvas'); off.width = canvasElement.width; off.height = canvasElement.height;
    const offCtx = off.getContext('2d'); offCtx.drawImage(videoElement, 0, 0, off.width, off.height);
    const imgData = offCtx.getImageData(L, T, W, H);
    const dst = mainCtx.getImageData(L, T, W, H);

    const sx = segW / canvasElement.width, sy = segH / canvasElement.height;
    for (let y=0;y<H;y++){
      const sy2 = Math.floor((T+y) * sy);
      if (sy2 < 0 || sy2 >= segH) continue;
      for (let x=0;x<W;x++){
        const sx2 = Math.floor((L+x) * sx);
        if (sx2 < 0 || sx2 >= segW) continue;
        const id = sy2 * segW + sx2;
        if (segData[id] === 1) {
          const i = (y*W + x)*4;
          dst.data[i]   = imgData.data[i];
          dst.data[i+1] = imgData.data[i+1];
          dst.data[i+2] = imgData.data[i+2];
          dst.data[i+3] = imgData.data[i+3];
        }
      }
    }
    mainCtx.putImageData(dst, L, T);
    drawWatermark(mainCtx);
  } catch(e) {
    drawWatermark(mainCtx);
  }
}

/* Snapshot helpers */
function triggerFlash() {
  if (flashOverlay) {
    flashOverlay.classList.add('active');
    setTimeout(()=>flashOverlay.classList.remove('active'), 180);
  }
}

async function takeSnapshot() {
  if (!smoothedLandmarks) { alert('Face not detected'); return; }
  await ensureWatermarkLoaded();
  triggerFlash();

  const snap = document.createElement('canvas'); 
  snap.width = canvasElement.width; 
  snap.height = canvasElement.height;
  const ctx = snap.getContext('2d'); 
  ctx.drawImage(videoElement, 0, 0, snap.width, snap.height);

  drawJewelrySmart(smoothedState, ctx, smoothedLandmarks, { 
    faceWidth: (0.5*canvasElement.width), 
    faceHeight:(0.7*canvasElement.height), 
    faceShape: smoothedState.faceShape 
  });
  if (lastPersonSegmentation && lastPersonSegmentation.data) 
    compositeHeadOcclusion(ctx, smoothedLandmarks, lastPersonSegmentation);
  else 
    drawWatermark(ctx);

  lastSnapshotDataURL = snap.toDataURL('image/png');
  const preview = document.getElementById('snapshot-preview');
  if (preview) {
    preview.src = lastSnapshotDataURL;
    const m = document.getElementById('snapshot-modal');
    if (m) m.style.display = 'block';
  }
}
function saveSnapshot() {
  if (!lastSnapshotDataURL) return;
  const a = document.createElement('a'); 
  a.href = lastSnapshotDataURL; 
  a.download = `jewelry-${Date.now()}.png`; 
  a.click();
}
async function shareSnapshot() {
  if (!navigator.share) { alert('Sharing not supported'); return; }
  const blob = await (await fetch(lastSnapshotDataURL)).blob();
  const file = new File([blob], 'look.png', { type: 'image/png' });
  await navigator.share({ files: [file] });
}
function closeSnapshotModal() {
  const m = document.getElementById('snapshot-modal');
  if (m) m.style.display = 'none';
}

/* Try-all & gallery */
let autoTryRunning = false, autoTryTimeout = null, autoTryIndex = 0, autoSnapshots = [];

function stopAutoTry(){
  autoTryRunning = false;
  if (autoTryTimeout) clearTimeout(autoTryTimeout);
  autoTryTimeout = null;
  try { 
    tryAllBtn.classList.remove('active'); 
    tryAllBtn.textContent = 'Try All'; 
  } catch(e){}

  if (autoSnapshots && autoSnapshots.length) {
    openGallery();
  }
}
function toggleTryAll(){ 
  if (autoTryRunning) stopAutoTry(); 
  else startAutoTry(); 
}

async function startAutoTry(){
  if (!currentType) { alert('Choose a category first'); return; }
  const list = buildImageList(currentType);
  if (!list.length) { alert('No items'); return; }

  autoSnapshots = []; 
  autoTryIndex = 0; 
  autoTryRunning = true;
  try { 
    tryAllBtn.classList.add('active'); 
    tryAllBtn.textContent = 'Stop'; 
  } catch(e){}

  const step = async () => {
    if (!autoTryRunning) return;
    const src = list[autoTryIndex];
    if (currentType.includes('earrings')) await changeEarring(src); 
    else await changeNecklace(src);

    await new Promise(r => setTimeout(r, 800));
    triggerFlash();

    if (smoothedLandmarks) {
      const snap = document.createElement('canvas'); 
      snap.width = canvasElement.width; 
      snap.height = canvasElement.height;
      const ctx = snap.getContext('2d'); 
      try { ctx.drawImage(videoElement, 0, 0, snap.width, snap.height); } catch(e) {}
      drawJewelrySmart(smoothedState, ctx, smoothedLandmarks, { 
        faceWidth: (0.5*canvasElement.width), 
        faceHeight:(0.7*canvasElement.height), 
        faceShape: smoothedState.faceShape 
      });
      if (lastPersonSegmentation && lastPersonSegmentation.data) 
        compositeHeadOcclusion(ctx, smoothedLandmarks, lastPersonSegmentation);
      else 
        drawWatermark(ctx);
      autoSnapshots.push(snap.toDataURL('image/png'));
    }

    autoTryIndex++;
    if (autoTryIndex >= list.length) {
      autoTryRunning = false;
      try { 
        tryAllBtn.classList.remove('active'); 
        tryAllBtn.textContent = 'Try All'; 
      } catch(e){}
      if (autoSnapshots.length) openGallery();
      return;
    }
    autoTryTimeout = setTimeout(step, 2000);
  };
  step();
}

/* Safer openGallery: only open if snapshots exist */
function openGallery(){
  if (!autoSnapshots || !autoSnapshots.length) return;
  if (!galleryThumbs) return;

  galleryThumbs.innerHTML = '';
  autoSnapshots.forEach((src,i) => {
    const img = document.createElement('img'); 
    img.src = src;
    img.onclick = () => setGalleryMain(i);
    galleryThumbs.appendChild(img);
  });
  setGalleryMain(0);
  const gm = document.getElementById('gallery-modal');
  if (gm) gm.style.display = 'flex';
  document.body.style.overflow = 'hidden';
}
function setGalleryMain(i){
  if (!galleryMain) return;
  galleryMain.src = autoSnapshots[i];
  const thumbs = galleryThumbs.querySelectorAll('img');
  thumbs.forEach((t,idx) => t.classList.toggle('active', idx === i));
}

/* ---- Robust gallery close / cleanup ---- */
function closeGalleryClean() {
  try {
    if (typeof stopAutoTry === 'function') stopAutoTry();
  } catch(e){}

  autoSnapshots = [];

  const gm = document.getElementById('gallery-modal');
  if (gm) {
    gm.style.display = 'none';
    gm.style.pointerEvents = 'auto';
  }

  document.body.style.overflow = '';
  try { window.focus(); } catch(e){}
}

const galleryCloseBtn = document.getElementById('gallery-close');
if (galleryCloseBtn) {
  try { galleryCloseBtn.removeEventListener && galleryCloseBtn.removeEventListener('click', closeGalleryClean); } catch(e){}
  galleryCloseBtn.addEventListener('click', closeGalleryClean);
}

/* download / share helpers */
async function downloadAllImages(){
  if (!autoSnapshots.length) return;
  const zip = new JSZip(), f = zip.folder('Looks');
  for (let i=0;i<autoSnapshots.length;i++){
    const b = autoSnapshots[i].split(',')[1];
    f.file(`look_${i+1}.png`, b, { base64: true });
  }
  const blob = await zip.generateAsync({ type: 'blob' });
  saveAs(blob, 'Looks.zip');
}
async function shareCurrentFromGallery(){
  if (!navigator.share) { alert('Share not supported'); return; }
  const blob = await (await fetch(galleryMain.src)).blob();
  const file = new File([blob], 'look.png', { type:'image/png' });
  await navigator.share({ files: [file] });
}

/* ===========================
   Asset UI: categories & thumbnails
   ===========================*/

function toggleCategory(category){
  const subPanel = document.getElementById('subcategory-buttons');
  if (subPanel) subPanel.style.display = 'flex';

  const subs = document.querySelectorAll('#subcategory-buttons button');
  subs.forEach(b => {
    const label = b.innerText.toLowerCase();
    b.style.display = label.includes(category) ? 'inline-block' : 'none';
  });

  const jopt = document.getElementById('jewelry-options'); 
  if (jopt) jopt.style.display = 'none';

  stopAutoTry();
}

/* how many images for this type */
function getCountForType(type){
  return IMAGE_COUNTS[type] || 0;
}

/* When subcategory is selected */
function selectJewelryType(type){
  currentType = type;

  const container = document.getElementById('jewelry-options');
  if (!container) return;

  container.innerHTML = '';
  container.style.display = 'flex';

  earringImg = null; 
  necklaceImg = null;

  stopAutoTry();

  const count = getCountForType(type);
  for (let i = 1; i <= count; i++){
    const src = `${type}/${i}.png`;
    const btn = document.createElement('button');
    const img = document.createElement('img');
    img.src = src;
    btn.appendChild(img);

    btn.onclick = () => {
      if (type.includes('earrings')) changeEarring(src);
      else changeNecklace(src);
    };

    container.appendChild(btn);
  }
}

/* Used by Try-All */
function buildImageList(type){
  const count = getCountForType(type);
  const list = [];
  for (let i = 1; i <= count; i++){
    list.push(`${type}/${i}.png`);
  }
  return list;
}

/* load earring / necklace images */
async function changeEarring(src){ earringImg = await loadImage(src); }
async function changeNecklace(src){ necklaceImg = await loadImage(src); }

/* watermark ensure */
function ensureWatermarkLoaded(){
  return new Promise(res => {
    if (watermarkImg.complete && watermarkImg.naturalWidth) res();
    else {
      watermarkImg.onload = () => res();
      watermarkImg.onerror = () => res();
    }
  });
}

/* info modal toggle */
function toggleInfoModal(){
  const m = document.getElementById('info-modal');
  if (m) m.style.display = (m.style.display === 'block') ? 'none' : 'block';
}

/* debug draw */
function drawDebugMarkers(){
  if (!smoothedState.leftEar) return;
  const ctx = canvasCtx;
  ctx.save();
  ctx.fillStyle = 'cyan';
  ctx.beginPath();
  ctx.arc(smoothedState.leftEar.x, smoothedState.leftEar.y, 6, 0, Math.PI*2);
  ctx.fill();
  ctx.fillText('L', smoothedState.leftEar.x + 8, smoothedState.leftEar.y);

  ctx.fillStyle = 'magenta';
  ctx.beginPath();
  ctx.arc(smoothedState.rightEar.x, smoothedState.rightEar.y, 6, 0, Math.PI*2);
  ctx.fill();
  ctx.fillText('R', smoothedState.rightEar.x + 8, smoothedState.rightEar.y);

  ctx.fillStyle = 'yellow';
  ctx.beginPath();
  ctx.arc(smoothedState.neckPoint.x, smoothedState.neckPoint.y, 6, 0, Math.PI*2);
  ctx.fill();
  ctx.fillText('N', smoothedState.neckPoint.x + 8, smoothedState.neckPoint.y);
  ctx.restore();
}

/* slider bindings (if present) */
if (earSizeRange.addEventListener) earSizeRange.addEventListener('input', () => {
  EAR_SIZE_FACTOR = parseFloat(earSizeRange.value);
  if (earSizeVal) earSizeVal.textContent = EAR_SIZE_FACTOR.toFixed(2);
});
if (neckYRange.addEventListener) neckYRange.addEventListener('input', () => {
  NECK_Y_OFFSET_FACTOR = parseFloat(neckYRange.value);
  if (neckYVal) neckYVal.textContent = NECK_Y_OFFSET_FACTOR.toFixed(2);
});
if (neckScaleRange.addEventListener) neckScaleRange.addEventListener('input', () => {
  NECK_SCALE_MULTIPLIER = parseFloat(neckScaleRange.value);
  if (neckScaleVal) neckScaleVal.textContent = NECK_SCALE_MULTIPLIER.toFixed(2);
});
if (posSmoothRange.addEventListener) posSmoothRange.addEventListener('input', () => {
  POS_SMOOTH = parseFloat(posSmoothRange.value);
  if (posSmoothVal) posSmoothVal.textContent = POS_SMOOTH.toFixed(2);
});
if (earSmoothRange.addEventListener) earSmoothRange.addEventListener('input', () => {
  EAR_DIST_SMOOTH = parseFloat(earSmoothRange.value);
  if (earSmoothVal) earSmoothVal.textContent = EAR_DIST_SMOOTH.toFixed(2);
});

if (debugToggle.addEventListener)
  debugToggle.addEventListener('click', () => debugToggle.classList.toggle('on') );

/* start BodyPix load early */
ensureBodyPixLoaded();

/* expose functions for HTML onclicks */
window.toggleCategory = toggleCategory;
window.selectJewelryType = selectJewelryType;
window.takeSnapshot = takeSnapshot;
window.saveSnapshot = saveSnapshot;
window.shareSnapshot = shareSnapshot;
window.closeSnapshotModal = closeSnapshotModal;
window.toggleTryAll = toggleTryAll;
window.downloadAllImages = downloadAllImages;
window.shareCurrentFromGallery = shareCurrentFromGallery;
window.toggleInfoModal = toggleInfoModal;

/* ===========================
   DISABLE RIGHT CLICK & DEV TOOLS
   ============================ */
document.addEventListener('contextmenu', (e) => e.preventDefault());
document.onkeydown = function(e) {
  if (e.keyCode === 123) return false; // F12
  if (e.ctrlKey && e.shiftKey && (e.keyCode === 73 || e.keyCode === 74 || e.keyCode === 67 || e.keyCode === 75)) return false;
  if (e.ctrlKey && e.keyCode === 85) return false; // Ctrl+U
};
