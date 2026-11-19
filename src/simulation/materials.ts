import * as THREE from 'three';

export interface ProceduralAssets {
  trackTexture: THREE.CanvasTexture;
  carbonTexture: THREE.CanvasTexture;
  brushedMetalTexture: THREE.CanvasTexture;
  hazardTexture: THREE.CanvasTexture;
  contactShadowTexture: THREE.CanvasTexture;
  dispose: () => void;
}

export function createProceduralAssets(maxAnisotropy: number): ProceduralAssets {
  // 1. Calibrated Laboratory Track Texture (tiled 2m x 4m section with meter lines)
  const trackCanvas = document.createElement('canvas');
  trackCanvas.width = 512;
  trackCanvas.height = 512;
  const tctx = trackCanvas.getContext('2d')!;
  tctx.fillStyle = '#121720';
  tctx.fillRect(0, 0, 512, 512);

  // Subtle surface speckle
  for (let i = 0; i < 2800; i++) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const v = 16 + Math.floor(Math.random() * 14);
    tctx.fillStyle = `rgb(${v}, ${v + 4}, ${v + 10})`;
    tctx.fillRect(x, y, 2, 2);
  }

  // Fine calibration sub-grid
  tctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
  tctx.lineWidth = 1;
  for (let i = 0; i <= 512; i += 64) {
    tctx.beginPath();
    tctx.moveTo(i, 0);
    tctx.lineTo(i, 512);
    tctx.stroke();

    tctx.beginPath();
    tctx.moveTo(0, i);
    tctx.lineTo(512, i);
    tctx.stroke();
  }

  // Centerline guide stripe
  tctx.fillStyle = 'rgba(245, 158, 11, 0.35)';
  tctx.fillRect(253, 0, 6, 512);

  // Lateral track lane rails
  tctx.fillStyle = 'rgba(56, 189, 248, 0.22)';
  tctx.fillRect(24, 0, 5, 512);
  tctx.fillRect(483, 0, 5, 512);

  // 1-meter horizontal calibration ticks
  tctx.fillStyle = 'rgba(226, 232, 240, 0.22)';
  tctx.fillRect(24, 0, 464, 3);
  tctx.fillRect(24, 256, 464, 2);

  const trackTexture = new THREE.CanvasTexture(trackCanvas);
  trackTexture.colorSpace = THREE.SRGBColorSpace;
  trackTexture.wrapS = THREE.RepeatWrapping;
  trackTexture.wrapT = THREE.RepeatWrapping;
  trackTexture.anisotropy = maxAnisotropy;

  // 2. Carbon Fiber Weave Texture
  const carbonCanvas = document.createElement('canvas');
  carbonCanvas.width = 128;
  carbonCanvas.height = 128;
  const cctx = carbonCanvas.getContext('2d')!;
  cctx.fillStyle = '#1b2029';
  cctx.fillRect(0, 0, 128, 128);
  for (let y = 0; y < 128; y += 8) {
    for (let x = 0; x < 128; x += 8) {
      const odd = ((x + y) / 8) % 2 === 0;
      cctx.fillStyle = odd ? '#262d38' : '#151a22';
      cctx.fillRect(x, y, 7, 7);
    }
  }
  const carbonTexture = new THREE.CanvasTexture(carbonCanvas);
  carbonTexture.colorSpace = THREE.SRGBColorSpace;
  carbonTexture.wrapS = THREE.RepeatWrapping;
  carbonTexture.wrapT = THREE.RepeatWrapping;
  carbonTexture.repeat.set(3, 3);
  carbonTexture.anisotropy = maxAnisotropy;

  // 3. Brushed Titanium Texture
  const metalCanvas = document.createElement('canvas');
  metalCanvas.width = 256;
  metalCanvas.height = 256;
  const mctx = metalCanvas.getContext('2d')!;
  mctx.fillStyle = '#cbd5e1';
  mctx.fillRect(0, 0, 256, 256);
  for (let y = 0; y < 256; y++) {
    const shade = 185 + Math.floor(Math.random() * 45);
    mctx.fillStyle = `rgb(${shade}, ${shade + 4}, ${shade + 10})`;
    mctx.fillRect(0, y, 256, 1);
  }
  const brushedMetalTexture = new THREE.CanvasTexture(metalCanvas);
  brushedMetalTexture.colorSpace = THREE.SRGBColorSpace;
  brushedMetalTexture.wrapS = THREE.RepeatWrapping;
  brushedMetalTexture.wrapT = THREE.RepeatWrapping;
  brushedMetalTexture.anisotropy = maxAnisotropy;

  // 4. Hazard Stripe Texture for Terrain Obstacles
  const hazCanvas = document.createElement('canvas');
  hazCanvas.width = 256;
  hazCanvas.height = 256;
  const hctx = hazCanvas.getContext('2d')!;
  hctx.fillStyle = '#1e2430';
  hctx.fillRect(0, 0, 256, 256);
  hctx.fillStyle = '#d97706';
  for (let i = -256; i < 512; i += 48) {
    hctx.beginPath();
    hctx.moveTo(i, 0);
    hctx.lineTo(i + 24, 0);
    hctx.lineTo(i + 24 + 256, 256);
    hctx.lineTo(i + 256, 256);
    hctx.closePath();
    hctx.fill();
  }
  const hazardTexture = new THREE.CanvasTexture(hazCanvas);
  hazardTexture.colorSpace = THREE.SRGBColorSpace;
  hazardTexture.wrapS = THREE.RepeatWrapping;
  hazardTexture.wrapT = THREE.RepeatWrapping;
  hazardTexture.anisotropy = maxAnisotropy;

  // 5. Soft Radial Contact Shadow Texture
  const shadowCanvas = document.createElement('canvas');
  shadowCanvas.width = 128;
  shadowCanvas.height = 128;
  const sctx = shadowCanvas.getContext('2d')!;
  const grad = sctx.createRadialGradient(64, 64, 4, 64, 64, 62);
  grad.addColorStop(0, 'rgba(0, 0, 0, 0.55)');
  grad.addColorStop(0.55, 'rgba(0, 0, 0, 0.22)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  sctx.fillStyle = grad;
  sctx.fillRect(0, 0, 128, 128);
  const contactShadowTexture = new THREE.CanvasTexture(shadowCanvas);

  return {
    trackTexture,
    carbonTexture,
    brushedMetalTexture,
    hazardTexture,
    contactShadowTexture,
    dispose: () => {
      trackTexture.dispose();
      carbonTexture.dispose();
      brushedMetalTexture.dispose();
      hazardTexture.dispose();
      contactShadowTexture.dispose();
    },
  };
}

export function createDistanceMarkerTexture(label: string, accent = false): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 96;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = accent ? '#78350f' : '#111827';
  ctx.fillRect(0, 0, 256, 96);

  ctx.strokeStyle = accent ? '#f59e0b' : '#38bdf8';
  ctx.lineWidth = 4;
  ctx.strokeRect(4, 4, 248, 88);

  ctx.fillStyle = accent ? '#fef3c7' : '#e2e8f0';
  ctx.font = 'bold 42px "Chakra Petch", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, 128, 49);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
