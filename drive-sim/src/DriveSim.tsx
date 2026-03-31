// @ts-nocheck
import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';

// ─── Types ────────────────────────────────────────────────────────────────────
interface CollisionObject {
  mesh: THREE.Object3D;
  type: 'building' | 'lamp' | 'pole' | 'vehicle' | 'pedestrian' | 'cyclist';
  radius?: number;
  isBox?: boolean;
  width?: number;
  depth?: number;
  height?: number;
  data?: PedestrianData | CyclistData | VehicleData;
}

interface VehicleData {
  mesh: THREE.Group;
  wheels: THREE.Mesh[];
  speed: number;
  direction: number;
  isHorizontal: boolean;
  lane: number;
  stoppedAtLight: boolean;
  stopTimer: number;
}

interface PedestrianData {
  mesh: THREE.Group;
  speed: number;
  direction: number;
  throwVelocity: THREE.Vector3;
  isThrown: boolean;
  blockX: number;
  blockZ: number;
  onCrossing: boolean;
}

interface CyclistData {
  mesh: THREE.Group;
  speed: number;
  direction: number;
  isHorizontal: boolean;
  throwVelocity: THREE.Vector3;
  isThrown: boolean;
  wheels: THREE.Mesh[];
}

interface TrafficLight {
  red: THREE.Mesh;
  yellow: THREE.Mesh;
  green: THREE.Mesh;
  timer: number;
  state: string;
  position: THREE.Vector3;
}

// ─── Audio Engine ─────────────────────────────────────────────────────────────
class AudioEngine {
  ctx: AudioContext | null = null;
  engineGain: GainNode | null = null;
  engineOsc: OscillatorNode | null = null;

  init() {
    try {
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.engineGain = this.ctx.createGain();
      this.engineGain.gain.value = 0;
      this.engineGain.connect(this.ctx.destination);

      this.engineOsc = this.ctx.createOscillator();
      this.engineOsc.type = 'sawtooth';
      this.engineOsc.frequency.value = 60;

      const filter = this.ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 400;
      this.engineOsc.connect(filter);
      filter.connect(this.engineGain);
      this.engineOsc.start();
    } catch (_) { }
  }

  setEngine(speed: number, maxSpeed: number) {
    if (!this.ctx || !this.engineGain || !this.engineOsc) return;
    const ratio = Math.abs(speed) / maxSpeed;
    this.engineGain.gain.setTargetAtTime(ratio * 0.08, this.ctx.currentTime, 0.1);
    this.engineOsc.frequency.setTargetAtTime(60 + ratio * 180, this.ctx.currentTime, 0.05);
  }

  playImpact(intensity: number) {
    if (!this.ctx) return;
    const buf = this.ctx.createBuffer(1, this.ctx.sampleRate * 0.3, this.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (data.length * 0.2)) * intensity;
    }
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    src.start();
  }

  resume() {
    this.ctx?.resume();
  }
}

// ─── Component ────────────────────────────────────────────────────────────────
const OpenWorldDrivingSim: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<AudioEngine>(new AudioEngine());
  const gameInitialized = useRef(false);

  const [speed, setSpeed] = useState(0);
  const [gear, setGear] = useState('P');
  const [score, setScore] = useState(0);
  const [violations, setViolations] = useState(0);
  const [damage, setDamage] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [carPos, setCarPos] = useState({ x: 0, z: 0 });
  const [timeOfDay, setTimeOfDay] = useState(0); // 0–1 (0=noon, 0.5=night)
  const [rpm, setRpm] = useState(0);

  // ─── Init / Teardown ────────────────────────────────────────────────────────
  const initGame = useCallback(() => {
    if (!mountRef.current) return;

    // ── Scene ────────────────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB);
    scene.fog = new THREE.Fog(0x87CEEB, 80, 350);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.5, 600);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
    } catch (e) {
      // WebGL not available — show error to user
      const msg = document.createElement('div');
      msg.style.cssText = 'position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:#111;color:#ff4444;font-family:monospace;font-size:18px;text-align:center;padding:40px;z-index:9999;';
      msg.innerHTML = `<div><h2 style="color:#ff6666;margin-bottom:16px;">⚠ WebGL Not Available</h2>
        <p style="color:#aaa;font-size:14px;max-width:500px;line-height:1.6;">Your browser cannot create a WebGL context.<br><br>
        <b>Fix:</b> Open <span style="color:#88ff88">chrome://settings/system</span> and enable<br>
        <b>"Use graphics acceleration when available"</b>,<br>then restart your browser.<br><br>
        Or try: <span style="color:#88ff88">chrome --enable-webgl --ignore-gpu-blocklist</span></p></div>`;
      mountRef.current.appendChild(msg);
      return;
    }

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.BasicShadowMap;
    mountRef.current.appendChild(renderer.domElement);

    // Init audio on first user click (avoids autoplay policy warning)
    const audio = audioRef.current;
    const initAudioOnce = () => { audio.init(); document.removeEventListener('click', initAudioOnce); };
    document.addEventListener('click', initAudioOnce);

    // ── Lighting ─────────────────────────────────────────────────────────────
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff5e0, 1.0);
    sunLight.position.set(50, 100, 50);
    sunLight.castShadow = true;
    sunLight.shadow.camera.left = -80;
    sunLight.shadow.camera.right = 80;
    sunLight.shadow.camera.top = 80;
    sunLight.shadow.camera.bottom = -80;
    sunLight.shadow.mapSize.width = 512;
    sunLight.shadow.mapSize.height = 512;
    scene.add(sunLight);

    // ── Ground ───────────────────────────────────────────────────────────────
    const groundGeo = new THREE.PlaneGeometry(1200, 1200);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x2d6a2d });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // ── Clouds (animated) ────────────────────────────────────────────────────
    const clouds: THREE.Mesh[] = [];
    const cloudMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
    for (let i = 0; i < 18; i++) {
      const cg = new THREE.Group();
      const nBlobs = 3 + Math.floor(Math.random() * 4);
      for (let b = 0; b < nBlobs; b++) {
        const blob = new THREE.Mesh(
          new THREE.SphereGeometry(4 + Math.random() * 6, 6, 6),
          cloudMat
        );
        blob.position.set((b - nBlobs / 2) * 5, Math.random() * 2, Math.random() * 3);
        blob.scale.y = 0.4 + Math.random() * 0.3;
        cg.add(blob);
      }
      cg.position.set((Math.random() - 0.5) * 600, 60 + Math.random() * 40, (Math.random() - 0.5) * 600);
      (cg as any).cloudSpeed = 0.02 + Math.random() * 0.04;
      scene.add(cg);
      clouds.push(cg as any);
    }

    // Track swaying trees for animation
    const swayTrees: THREE.Object3D[] = [];

    // ── Collision objects ────────────────────────────────────────────────────
    const collisionObjects: CollisionObject[] = [];

    // ── Roads ────────────────────────────────────────────────────────────────
    const roadMat = new THREE.MeshLambertMaterial({ color: 0x2a2a2a });
    const lineMat = new THREE.MeshBasicMaterial({ color: 0xEEEEEE });
    const sidewalkMat = new THREE.MeshLambertMaterial({ color: 0xBBBBBB });
    const zebraMat = new THREE.MeshBasicMaterial({ color: 0xFFFFFF });

    const dashGeoH = new THREE.BoxGeometry(4, 0.15, 0.3);
    const dashGeoV = new THREE.BoxGeometry(0.3, 0.15, 4);

    for (let i = -6; i <= 6; i++) {
      // Road
      const rH = new THREE.Mesh(new THREE.BoxGeometry(600, 0.1, 12), roadMat);
      rH.position.set(0, 0.05, i * 50);
      scene.add(rH);

      const rV = new THREE.Mesh(new THREE.BoxGeometry(12, 0.1, 600), roadMat);
      rV.position.set(i * 50, 0.05, 0);
      scene.add(rV);

      // Dashes (every 3rd for perf)
      for (let j = -8; j <= 8; j++) {
        const dH = new THREE.Mesh(dashGeoH, lineMat);
        dH.position.set(j * 36, 0.1, i * 50);
        scene.add(dH);

        const dV = new THREE.Mesh(dashGeoV, lineMat);
        dV.position.set(i * 50, 0.1, j * 36);
        scene.add(dV);
      }

      // Sidewalks
      for (const side of [-1, 1]) {
        const swH = new THREE.Mesh(new THREE.BoxGeometry(600, 0.2, 3), sidewalkMat);
        swH.position.set(0, 0.1, i * 50 + side * 7.5);
        scene.add(swH);

        const swV = new THREE.Mesh(new THREE.BoxGeometry(3, 0.2, 600), sidewalkMat);
        swV.position.set(i * 50 + side * 7.5, 0.1, 0);
        scene.add(swV);
      }
    }

    // Zebra crossings (every other intersection, fewer stripes)
    const zebraGeoH = new THREE.BoxGeometry(12, 0.15, 1);
    const zebraGeoV = new THREE.BoxGeometry(1, 0.15, 12);
    for (let i = -4; i <= 4; i += 2) {
      for (let j = -4; j <= 4; j += 2) {
        for (let z = 0; z < 3; z++) {
          const zH = new THREE.Mesh(zebraGeoH, zebraMat);
          zH.position.set(i * 50, 0.11, j * 50 + z * 4 - 4);
          scene.add(zH);

          const zV = new THREE.Mesh(zebraGeoV, zebraMat);
          zV.position.set(i * 50 + z * 4 - 4, 0.11, j * 50);
          scene.add(zV);
        }
      }
    }

    // ── Helper: check if position is on/near a road ────────────────────────
    const isOnRoad = (x: number, z: number, margin: number) => {
      for (let r = -6; r <= 6; r++) {
        if (Math.abs(x - r * 50) < margin) return true;
        if (Math.abs(z - r * 50) < margin) return true;
      }
      return false;
    };

    // ── Shared materials ───────────────────────────────────────────────────
    const winGeo = new THREE.BoxGeometry(1.5, 1.5, 0.15);
    const winLit = new THREE.MeshBasicMaterial({ color: 0xFFFF88 });
    const winDark = new THREE.MeshBasicMaterial({ color: 0x222233 });
    const winBlue = new THREE.MeshStandardMaterial({ color: 0x6688bb, metalness: 0.8, roughness: 0.1, transparent: true, opacity: 0.7 });
    const concreteMat = new THREE.MeshLambertMaterial({ color: 0x999999 });
    const darkMetalMat = new THREE.MeshLambertMaterial({ color: 0x333333 });
    const woodMat = new THREE.MeshLambertMaterial({ color: 0x8B6914 });
    const redMat = new THREE.MeshLambertMaterial({ color: 0xCC3333 });
    const greenHedgeMat = new THREE.MeshLambertMaterial({ color: 0x2D5A1E });

    // ── Building archetypes ─────────────────────────────────────────────────
    const buildBlocks = (blockCX: number, blockCZ: number) => {
      const rand = Math.random();

      if (rand < 0.25) {
        // ── SHOP ROW: low buildings with awnings ──
        const numShops = 2 + Math.floor(Math.random() * 2);
        const shopW = 8 + Math.random() * 4;
        const shopD = 8;
        const shopH = 5 + Math.random() * 3;
        const shopColors = [0xD4A574, 0xC4956A, 0xB8860B, 0xCD853F, 0xDEB887, 0xE8C89E];
        const awningColors = [0xCC3333, 0x2255AA, 0x33AA55, 0xDD8800, 0x8833AA];
        for (let s = 0; s < numShops; s++) {
          const sx = blockCX + (s - numShops / 2) * (shopW + 1);
          const sz = blockCZ;
          const maxOff = Math.max(0, (50 - 18 - shopD) / 2);
          const bz = sz + (Math.random() - 0.5) * maxOff;
          if (isOnRoad(sx, bz, 10)) continue;
          const shop = new THREE.Mesh(
            new THREE.BoxGeometry(shopW, shopH, shopD),
            new THREE.MeshLambertMaterial({ color: shopColors[Math.floor(Math.random() * shopColors.length)] })
          );
          shop.position.set(sx, shopH / 2, bz);
          shop.castShadow = true;
          scene.add(shop);
          collisionObjects.push({ mesh: shop, type: 'building', isBox: true, width: shopW, depth: shopD, height: shopH });

          // Awning
          const awning = new THREE.Mesh(
            new THREE.BoxGeometry(shopW + 1, 0.15, 2),
            new THREE.MeshLambertMaterial({ color: awningColors[Math.floor(Math.random() * awningColors.length)] })
          );
          awning.position.set(sx, shopH - 0.5, bz + shopD / 2 + 1);
          scene.add(awning);

          // Door
          const door = new THREE.Mesh(new THREE.BoxGeometry(1.2, 2.2, 0.15), new THREE.MeshLambertMaterial({ color: 0x4A3520 }));
          door.position.set(sx, 1.1, bz + shopD / 2 + 0.1);
          scene.add(door);

          // Shop window
          const sw = new THREE.Mesh(new THREE.BoxGeometry(shopW * 0.6, 2, 0.12), winBlue);
          sw.position.set(sx + 1.5, 2.2, bz + shopD / 2 + 0.1);
          scene.add(sw);
        }
      } else if (rand < 0.5) {
        // ── TALL APARTMENT ──
        const h = 25 + Math.random() * 30;
        const w = 10 + Math.random() * 6;
        const d = 10 + Math.random() * 6;
        const maxOff = Math.max(0, (50 - 18 - Math.max(w, d)) / 2);
        const bx = blockCX + (Math.random() - 0.5) * maxOff * 2;
        const bz = blockCZ + (Math.random() - 0.5) * maxOff * 2;
        const aptColors = [0x8899AA, 0x778899, 0x6B7B8D, 0x9DA8B2, 0xA0ADB8, 0x7A8A9A];
        const building = new THREE.Mesh(
          new THREE.BoxGeometry(w, h, d),
          new THREE.MeshLambertMaterial({ color: aptColors[Math.floor(Math.random() * aptColors.length)] })
        );
        building.position.set(bx, h / 2, bz);
        building.castShadow = true;
        scene.add(building);
        collisionObjects.push({ mesh: building, type: 'building', isBox: true, width: w, depth: d, height: h });

        // Window grid (3 cols × up to 5 rows)
        for (let side = -1; side <= 1; side += 2) {
          for (let wx = 0; wx < 3; wx++) {
            for (let wy = 0; wy < Math.min(5, Math.floor(h / 6)); wy++) {
              const win = new THREE.Mesh(winGeo, Math.random() > 0.3 ? winLit : winDark);
              win.position.set(bx + (wx - 1) * 3.5, wy * 5 + 4, bz + side * (d / 2 + 0.1));
              scene.add(win);
            }
          }
        }

        // Rooftop water tank
        if (Math.random() > 0.4) {
          const tank = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 2.5, 6), concreteMat);
          tank.position.set(bx + 2, h + 1.25, bz);
          scene.add(tank);
        }
        // AC units on roof
        for (let a = 0; a < 2; a++) {
          const ac = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.8, 1), darkMetalMat);
          ac.position.set(bx - 2 + a * 2, h + 0.4, bz - 2);
          scene.add(ac);
        }

      } else if (rand < 0.65) {
        // ── RESIDENTIAL HOUSE ──
        const hW = 7 + Math.random() * 4;
        const hD = 8 + Math.random() * 3;
        const hH = 4 + Math.random() * 2;
        const maxOff = Math.max(0, (50 - 18 - Math.max(hW, hD)) / 2);
        const bx = blockCX + (Math.random() - 0.5) * maxOff * 2;
        const bz = blockCZ + (Math.random() - 0.5) * maxOff * 2;
        const houseColors = [0xFFF8DC, 0xFAEBD7, 0xF5DEB3, 0xE6D5B8, 0xD2B48C, 0xF0E68C];
        const house = new THREE.Mesh(
          new THREE.BoxGeometry(hW, hH, hD),
          new THREE.MeshLambertMaterial({ color: houseColors[Math.floor(Math.random() * houseColors.length)] })
        );
        house.position.set(bx, hH / 2, bz);
        house.castShadow = true;
        scene.add(house);
        collisionObjects.push({ mesh: house, type: 'building', isBox: true, width: hW, depth: hD, height: hH });

        // Pitched roof
        const roofGeo = new THREE.ConeGeometry(Math.max(hW, hD) * 0.6, 3, 4);
        const roof = new THREE.Mesh(roofGeo, new THREE.MeshLambertMaterial({ color: 0x8B4513 }));
        roof.position.set(bx, hH + 1.5, bz);
        roof.rotation.y = Math.PI / 4;
        scene.add(roof);

        // Front door + windows
        const fd = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1.8, 0.12), new THREE.MeshLambertMaterial({ color: 0x5C3A1E }));
        fd.position.set(bx, 0.9, bz + hD / 2 + 0.08);
        scene.add(fd);
        for (const wx of [-2, 2]) {
          const hw = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 0.12), winBlue);
          hw.position.set(bx + wx, 2.5, bz + hD / 2 + 0.08);
          scene.add(hw);
        }

        // Garden fence
        if (Math.random() > 0.5) {
          for (let f = -3; f <= 3; f++) {
            const fPost = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1, 0.1), woodMat);
            fPost.position.set(bx + f * 1.5, 0.5, bz + hD / 2 + 3);
            scene.add(fPost);
          }
          const fRail = new THREE.Mesh(new THREE.BoxGeometry(9, 0.08, 0.08), woodMat);
          fRail.position.set(bx, 0.8, bz + hD / 2 + 3);
          scene.add(fRail);
        }

      } else if (rand < 0.8) {
        // ── WAREHOUSE / COMMERCIAL ──
        const wH = 8 + Math.random() * 4;
        const wW = 14 + Math.random() * 8;
        const wD = 12 + Math.random() * 6;
        const maxOff = Math.max(0, (50 - 18 - Math.max(wW, wD)) / 2);
        const bx = blockCX + (Math.random() - 0.5) * maxOff;
        const bz = blockCZ + (Math.random() - 0.5) * maxOff;
        const whColors = [0x8B8682, 0x708090, 0x696969, 0x808080];
        const wh = new THREE.Mesh(
          new THREE.BoxGeometry(wW, wH, wD),
          new THREE.MeshLambertMaterial({ color: whColors[Math.floor(Math.random() * whColors.length)] })
        );
        wh.position.set(bx, wH / 2, bz);
        wh.castShadow = true;
        scene.add(wh);
        collisionObjects.push({ mesh: wh, type: 'building', isBox: true, width: wW, depth: wD, height: wH });

        // Loading bay door
        const bayDoor = new THREE.Mesh(new THREE.BoxGeometry(4, 5, 0.15), new THREE.MeshLambertMaterial({ color: 0x555555 }));
        bayDoor.position.set(bx, 2.5, bz + wD / 2 + 0.1);
        scene.add(bayDoor);

      } else {
        // ── GLASS OFFICE ──
        const oH = 20 + Math.random() * 25;
        const oW = 10 + Math.random() * 6;
        const oD = 10 + Math.random() * 6;
        const maxOff = Math.max(0, (50 - 18 - Math.max(oW, oD)) / 2);
        const bx = blockCX + (Math.random() - 0.5) * maxOff * 2;
        const bz = blockCZ + (Math.random() - 0.5) * maxOff * 2;
        const office = new THREE.Mesh(
          new THREE.BoxGeometry(oW, oH, oD),
          new THREE.MeshStandardMaterial({ color: 0x4477AA, metalness: 0.6, roughness: 0.2, transparent: true, opacity: 0.85 })
        );
        office.position.set(bx, oH / 2, bz);
        office.castShadow = true;
        scene.add(office);
        collisionObjects.push({ mesh: office, type: 'building', isBox: true, width: oW, depth: oD, height: oH });

        // Glass panel lines
        for (let fl = 0; fl < Math.min(6, Math.floor(oH / 4)); fl++) {
          const stripe = new THREE.Mesh(new THREE.BoxGeometry(oW + 0.1, 0.08, oD + 0.1), darkMetalMat);
          stripe.position.set(bx, fl * 4 + 4, bz);
          scene.add(stripe);
        }
      }
    };

    // ── Place buildings in each block ────────────────────────────────────────
    for (let i = -5; i <= 4; i++) {
      for (let j = -5; j <= 4; j++) {
        const blockCX = i * 50 + 25;
        const blockCZ = j * 50 + 25;
        if (Math.abs(blockCX) < 30 && Math.abs(blockCZ) < 30) continue;
        if (Math.random() > 0.2) buildBlocks(blockCX, blockCZ);

        // ── Trees (varied types) ──
        const nTrees = Math.floor(Math.random() * 3);
        for (let t = 0; t < nTrees; t++) {
          const treeX = blockCX + (Math.random() - 0.5) * 20;
          const treeZ = blockCZ + (Math.random() - 0.5) * 20;
          if (isOnRoad(treeX, treeZ, 9)) continue;
          const treeGrp = new THREE.Group();
          const trunkH = 2 + Math.random() * 2;
          const trunk = new THREE.Mesh(
            new THREE.CylinderGeometry(0.2, 0.35, trunkH, 5),
            new THREE.MeshLambertMaterial({ color: 0x5C3A1E })
          );
          trunk.position.set(0, trunkH / 2, 0);
          treeGrp.add(trunk);
          const treeCol = [0x1B7A1B, 0x228B22, 0x2E8B57, 0x3CB371, 0x006400, 0x556B2F][Math.floor(Math.random() * 6)];
          const treeType = Math.random();
          if (treeType < 0.4) {
            // Round tree
            const canopy = new THREE.Mesh(
              new THREE.SphereGeometry(1.5 + Math.random() * 1.5, 6, 6),
              new THREE.MeshLambertMaterial({ color: treeCol })
            );
            canopy.position.set(0, trunkH + 1.5, 0);
            treeGrp.add(canopy);
          } else if (treeType < 0.7) {
            // Cone/pine tree
            const pine = new THREE.Mesh(
              new THREE.ConeGeometry(1.5 + Math.random(), 4 + Math.random() * 2, 6),
              new THREE.MeshLambertMaterial({ color: treeCol })
            );
            pine.position.set(0, trunkH + 2, 0);
            treeGrp.add(pine);
          } else {
            // Multi-blob tree
            for (let b = 0; b < 3; b++) {
              const blob = new THREE.Mesh(
                new THREE.SphereGeometry(1 + Math.random(), 5, 5),
                new THREE.MeshLambertMaterial({ color: treeCol })
              );
              blob.position.set((Math.random() - 0.5) * 1.5, trunkH + 1 + b * 0.8, (Math.random() - 0.5) * 1.5);
              treeGrp.add(blob);
            }
          }
          treeGrp.position.set(treeX, 0, treeZ);
          scene.add(treeGrp);
          swayTrees.push(treeGrp);
        }

        // ── Hedges along sidewalks ──
        if (Math.random() > 0.7) {
          const hx = blockCX + (Math.random() > 0.5 ? -12 : 12);
          const hz = blockCZ;
          if (!isOnRoad(hx, hz, 8)) {
            const hedge = new THREE.Mesh(
              new THREE.BoxGeometry(8 + Math.random() * 6, 1.2, 1),
              greenHedgeMat
            );
            hedge.position.set(hx, 0.6, hz);
            scene.add(hedge);
          }
        }
      }
    }

    // ── Street furniture (benches, bollards, trash cans, bus stops) ──────────
    const benchSeatGeo = new THREE.BoxGeometry(2, 0.12, 0.6);
    const benchLegGeo = new THREE.BoxGeometry(0.1, 0.5, 0.5);
    const bollardGeo = new THREE.CylinderGeometry(0.12, 0.14, 0.8, 6);
    const trashGeo = new THREE.CylinderGeometry(0.3, 0.35, 0.9, 6);
    const hydrantGeo = new THREE.CylinderGeometry(0.15, 0.2, 0.7, 6);

    for (let i = -4; i <= 4; i += 1) {
      for (let j = -4; j <= 4; j += 1) {
        const ix = i * 50;
        const jz = j * 50;

        // Benches on sidewalks (sparse)
        if (Math.random() > 0.75) {
          const bSide = Math.random() > 0.5 ? 9 : -9;
          const bx = ix + (Math.random() - 0.5) * 20;
          const bz = jz + bSide;
          const seat = new THREE.Mesh(benchSeatGeo, woodMat);
          seat.position.set(bx, 0.55, bz);
          scene.add(seat);
          const leg1 = new THREE.Mesh(benchLegGeo, darkMetalMat);
          leg1.position.set(bx - 0.7, 0.25, bz);
          scene.add(leg1);
          const leg2 = new THREE.Mesh(benchLegGeo, darkMetalMat);
          leg2.position.set(bx + 0.7, 0.25, bz);
          scene.add(leg2);
          // Backrest
          const back = new THREE.Mesh(new THREE.BoxGeometry(2, 0.6, 0.08), woodMat);
          back.position.set(bx, 0.9, bz - 0.25);
          scene.add(back);
        }

        // Bollards at intersections
        if (Math.abs(i) <= 4 && Math.abs(j) <= 4 && (i + j) % 2 === 0 && Math.random() > 0.5) {
          for (let b = 0; b < 3; b++) {
            const bol = new THREE.Mesh(bollardGeo, new THREE.MeshLambertMaterial({ color: 0x555555 }));
            bol.position.set(ix + 8 + b * 1.2, 0.4, jz + 8);
            scene.add(bol);
          }
        }

        // Trash cans
        if (Math.random() > 0.7) {
          const trash = new THREE.Mesh(trashGeo, new THREE.MeshLambertMaterial({ color: 0x2E8B57 }));
          trash.position.set(ix + (Math.random() > 0.5 ? 9 : -9), 0.45, jz + (Math.random() - 0.5) * 30);
          scene.add(trash);
        }

        // Fire hydrants
        if (Math.random() > 0.85) {
          const hydrant = new THREE.Mesh(hydrantGeo, redMat);
          hydrant.position.set(ix + 9.2, 0.35, jz + (Math.random() - 0.5) * 20);
          scene.add(hydrant);
        }

        // Bus stops (rare)
        if (Math.random() > 0.92) {
          const bsX = ix + 9;
          const bsZ = jz + (Math.random() - 0.5) * 10;
          // Shelter poles
          for (const off of [-1.5, 1.5]) {
            const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 3, 4), darkMetalMat);
            pole.position.set(bsX, 1.5, bsZ + off);
            scene.add(pole);
          }
          // Roof
          const bsRoof = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.1, 3.5), new THREE.MeshLambertMaterial({ color: 0x3388BB, transparent: true, opacity: 0.6 }));
          bsRoof.position.set(bsX, 3, bsZ);
          scene.add(bsRoof);
          // Bench inside
          const bsBench = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.1, 2.5), darkMetalMat);
          bsBench.position.set(bsX, 0.55, bsZ);
          scene.add(bsBench);
        }
      }
    }

    // ── Street lamps ─────────────────────────────────────────────────────────
    const lampPoleMat = new THREE.MeshLambertMaterial({ color: 0x444444 });
    const lampPoleGeo = new THREE.CylinderGeometry(0.12, 0.12, 8, 6);
    const lampHeadGeo = new THREE.SphereGeometry(0.45, 6, 6);
    const lampHeadMat = new THREE.MeshBasicMaterial({ color: 0xFFFF99 });
    for (let i = -4; i <= 4; i += 2) {
      for (let j = -4; j <= 4; j += 2) {
        // Place lamps on sidewalk corners (9.5 off road center = sidewalk edge)
        const lx = i * 50 + 9.5;
        const lz = j * 50 + 9.5;
        const pole = new THREE.Mesh(lampPoleGeo, lampPoleMat);
        pole.position.set(lx, 4, lz);
        scene.add(pole);

        const head = new THREE.Mesh(lampHeadGeo, lampHeadMat);
        head.position.set(lx, 8, lz);
        scene.add(head);

        const pt = new THREE.PointLight(0xFFFF88, 1.0, 30);
        pt.position.set(lx, 8, lz);
        scene.add(pt);

        collisionObjects.push({ mesh: pole, type: 'lamp', radius: 0.25 });
      }
    }

    // ── Traffic lights ───────────────────────────────────────────────────────
    const trafficLights: TrafficLight[] = [];
    const tlPoleMat = new THREE.MeshLambertMaterial({ color: 0x222222 });
    const tlPoleGeo = new THREE.CylinderGeometry(0.18, 0.18, 6, 6);
    const tlBoxGeo = new THREE.BoxGeometry(0.8, 2.4, 0.6);
    const tlBoxMat = new THREE.MeshLambertMaterial({ color: 0x111111 });
    const tlSphereGeo = new THREE.SphereGeometry(0.28, 6, 6);
    for (let i = -4; i <= 4; i += 2) {
      for (let j = -4; j <= 4; j += 2) {
        // Place traffic lights on sidewalk edges (9.2 off center = safe from road)
        const tlx = i * 50 + 9.2;
        const tlz = j * 50 + 9.2;
        const pole = new THREE.Mesh(tlPoleGeo, tlPoleMat);
        pole.position.set(tlx, 3, tlz);
        scene.add(pole);

        const box = new THREE.Mesh(tlBoxGeo, tlBoxMat);
        box.position.set(tlx, 5.2, tlz);
        scene.add(box);

        const mkLight = (color: number, y: number): THREE.Mesh => {
          const m = new THREE.Mesh(
            tlSphereGeo,
            new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0 })
          );
          m.position.set(tlx, y, tlz + 0.32);
          scene.add(m);
          return m;
        };

        const red = mkLight(0xFF2222, 5.7);
        const yellow = mkLight(0xFFDD00, 5.2);
        const green = mkLight(0x00EE44, 4.7);
        green.material.emissiveIntensity = 1;

        collisionObjects.push({ mesh: pole, type: 'pole', radius: 0.3 });

        const phaseOffset = (i + j) % 2 === 0 ? 0 : 198;
        trafficLights.push({ red, yellow, green, timer: phaseOffset, state: 'green', position: new THREE.Vector3(tlx, 0, tlz) });
      }
    }

    // ── Player car ───────────────────────────────────────────────────────────
    const carGroup = new THREE.Group();

    const carBodyMat = new THREE.MeshStandardMaterial({ color: 0xE83030, metalness: 0.7, roughness: 0.3 });
    const carBody = new THREE.Mesh(new THREE.BoxGeometry(2, 1, 4), carBodyMat);
    carBody.position.y = 0.5;
    carBody.castShadow = true;
    carGroup.add(carBody);

    const carTop = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.8, 2), carBodyMat);
    carTop.position.set(0, 1.3, -0.3);
    carTop.castShadow = true;
    carGroup.add(carTop);

    const bumperMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, metalness: 0.9 });
    const frontBumper = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.3, 0.3), bumperMat);
    frontBumper.position.set(0, 0.3, 2.15);
    carGroup.add(frontBumper);

    const backBumper = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.3, 0.3), bumperMat);
    backBumper.position.set(0, 0.3, -2.15);
    carGroup.add(backBumper);

    const winMat = new THREE.MeshStandardMaterial({ color: 0x88aacc, transparent: true, opacity: 0.55, metalness: 0.95, roughness: 0.05 });
    const fWin = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.6, 0.1), winMat);
    fWin.position.set(0, 1.3, 0.7);
    carGroup.add(fWin);

    const rWin = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.6, 0.1), winMat);
    rWin.position.set(0, 1.3, -1.25);
    carGroup.add(rWin);

    const wheelGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 8);
    const wheelMat = new THREE.MeshLambertMaterial({ color: 0x0d0d0d });
    const wheels: THREE.Mesh[] = [];
    for (const pos of [[-1, 0.4, 1.2], [1, 0.4, 1.2], [-1, 0.4, -1.2], [1, 0.4, -1.2]] as [number, number, number][]) {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.rotation.z = Math.PI / 2;
      w.position.set(...pos);
      carGroup.add(w);
      wheels.push(w);
    }

    const hlMat = new THREE.MeshStandardMaterial({ color: 0xFFFFFF, emissive: 0xFFFFEE, emissiveIntensity: 1 });
    const tlMat = new THREE.MeshStandardMaterial({ color: 0xFF0000, emissive: 0xFF0000, emissiveIntensity: 0.7 });

    for (const side of [-1, 1]) {
      const hl = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8), hlMat);
      hl.position.set(side * 0.7, 0.5, 2.1);
      carGroup.add(hl);

      const tl = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), tlMat);
      tl.position.set(side * 0.7, 0.5, -2.1);
      carGroup.add(tl);
    }

    scene.add(carGroup);

    // ── Other vehicles (diverse types) ────────────────────────────────────────
    const otherVehicles: VehicleData[] = [];
    const vColors = [0x2244CC, 0x22AA44, 0xCCCC22, 0xCC22CC, 0x22CCCC, 0xFF8800, 0xFFFFFF, 0x333333, 0x884422, 0xAA2222, 0x444488, 0xDDDDDD];

    // Vehicle type definitions: [bodyW, bodyH, bodyL, topW, topH, topL, topOffZ, wheelR, colRadius]
    const vTypes = [
      { name: 'sedan', bW: 2, bH: 1, bL: 4, tW: 1.8, tH: 0.8, tL: 2, tZ: -0.3, wR: 0.4, cR: 2.2 },
      { name: 'suv', bW: 2.2, bH: 1.3, bL: 4.5, tW: 2.0, tH: 1.0, tL: 2.8, tZ: -0.2, wR: 0.5, cR: 2.5 },
      { name: 'van', bW: 2.1, bH: 1.8, bL: 4.8, tW: 2.0, tH: 1.4, tL: 3.5, tZ: -0.5, wR: 0.45, cR: 2.6 },
      { name: 'truck', bW: 2.4, bH: 1.6, bL: 6, tW: 2.2, tH: 1.2, tL: 2, tZ: 1.5, wR: 0.55, cR: 3.2 },
    ];

    for (let i = 0; i < 30; i++) {
      const vg = new THREE.Group();
      const col = vColors[Math.floor(Math.random() * vColors.length)];
      const vMat = new THREE.MeshLambertMaterial({ color: col });
      const vt_idx = Math.random();
      const vType = vt_idx < 0.4 ? vTypes[0] : vt_idx < 0.65 ? vTypes[1] : vt_idx < 0.85 ? vTypes[2] : vTypes[3];

      const vb = new THREE.Mesh(new THREE.BoxGeometry(vType.bW, vType.bH, vType.bL), vMat);
      vb.position.y = vType.bH / 2 + 0.05;
      vg.add(vb);

      const vt = new THREE.Mesh(new THREE.BoxGeometry(vType.tW, vType.tH, vType.tL), vMat);
      vt.position.set(0, vType.bH + vType.tH / 2 + 0.05, vType.tZ);
      vg.add(vt);

      // Windshield
      const wsH = vType.tH * 0.7;
      const ws = new THREE.Mesh(new THREE.BoxGeometry(vType.tW * 0.85, wsH, 0.08),
        new THREE.MeshStandardMaterial({ color: 0x88aacc, transparent: true, opacity: 0.5, metalness: 0.9 }));
      ws.position.set(0, vType.bH + wsH / 2 + 0.1, vType.tZ + vType.tL / 2 + 0.04);
      vg.add(ws);

      // Headlights + taillights
      const hlMatV = new THREE.MeshBasicMaterial({ color: 0xFFFFCC });
      const tlMatV = new THREE.MeshBasicMaterial({ color: 0xFF2222 });
      for (const side of [-1, 1]) {
        const hl = new THREE.Mesh(new THREE.SphereGeometry(0.12, 4, 4), hlMatV);
        hl.position.set(side * (vType.bW / 2 - 0.15), vType.bH * 0.4, vType.bL / 2);
        vg.add(hl);
        const tl = new THREE.Mesh(new THREE.SphereGeometry(0.1, 4, 4), tlMatV);
        tl.position.set(side * (vType.bW / 2 - 0.15), vType.bH * 0.4, -vType.bL / 2);
        vg.add(tl);
      }

      // Truck cargo box
      if (vType.name === 'truck') {
        const cargo = new THREE.Mesh(new THREE.BoxGeometry(2.2, 2.2, 3),
          new THREE.MeshLambertMaterial({ color: 0x666666 }));
        cargo.position.set(0, 1.8, -1.5);
        vg.add(cargo);
      }

      const vWheels: THREE.Mesh[] = [];
      const wSpread = vType.bW / 2 + 0.1;
      const wFront = vType.bL * 0.3;
      const wRear = -vType.bL * 0.3;
      for (const pos of [[-wSpread, vType.wR, wFront], [wSpread, vType.wR, wFront],
      [-wSpread, vType.wR, wRear], [wSpread, vType.wR, wRear]] as [number, number, number][]) {
        const vw = new THREE.Mesh(new THREE.CylinderGeometry(vType.wR, vType.wR, 0.3, 8), wheelMat);
        vw.rotation.z = Math.PI / 2;
        vw.position.set(...pos);
        vg.add(vw);
        vWheels.push(vw);
      }

      const lane = Math.floor(Math.random() * 10) - 5;
      const isH = Math.random() > 0.5;
      const dir = Math.random() > 0.5 ? 1 : -1;
      vg.position.set(
        isH ? (Math.random() - 0.5) * 480 : lane * 50 + (dir > 0 ? 3 : -3),
        0,
        isH ? lane * 50 + (dir > 0 ? 3 : -3) : (Math.random() - 0.5) * 480
      );
      vg.rotation.y = isH ? (dir > 0 ? Math.PI / 2 : -Math.PI / 2) : (dir > 0 ? 0 : Math.PI);

      scene.add(vg);
      // Trucks are slower, sedans faster
      const baseSpeed = vType.name === 'truck' ? 0.12 : vType.name === 'van' ? 0.18 : 0.2;
      const speedRange = vType.name === 'truck' ? 0.25 : vType.name === 'suv' ? 0.45 : 0.5;
      const vData: VehicleData = { mesh: vg, wheels: vWheels, speed: baseSpeed + Math.random() * speedRange, direction: dir, isHorizontal: isH, lane, stoppedAtLight: false, stopTimer: 0 };
      otherVehicles.push(vData);
      collisionObjects.push({ mesh: vg, type: 'vehicle', radius: vType.cR, data: vData });
    }

    // ── Pedestrians ──────────────────────────────────────────────────────────
    const pedestrians: PedestrianData[] = [];
    const sharedPedBodyGeo = new THREE.CylinderGeometry(0.3, 0.3, 1.2, 5);
    const sharedPedHeadGeo = new THREE.SphereGeometry(0.25, 5, 5);
    const sharedPedArmGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.8, 4);
    const skinMat = new THREE.MeshLambertMaterial({ color: 0xFFCBA4 });

    for (let i = 0; i < 30; i++) {
      const pg = new THREE.Group();
      const bodyCol = Math.floor(Math.random() * 0xffffff);
      const bMat = new THREE.MeshLambertMaterial({ color: bodyCol });

      const pb = new THREE.Mesh(sharedPedBodyGeo, bMat); pb.position.y = 0.9; pg.add(pb);
      const ph = new THREE.Mesh(sharedPedHeadGeo, skinMat); ph.position.y = 1.75; pg.add(ph);
      const al = new THREE.Mesh(sharedPedArmGeo, bMat); al.position.set(-0.4, 1, 0); al.rotation.z = 0.3; pg.add(al);
      const ar = new THREE.Mesh(sharedPedArmGeo, bMat); ar.position.set(0.4, 1, 0); ar.rotation.z = -0.3; pg.add(ar);

      const bx = Math.floor(Math.random() * 9) - 4;
      const bz = Math.floor(Math.random() * 9) - 4;
      const onC = Math.random() > 0.6;
      const side = Math.random() > 0.5 ? 1 : -1;
      const isHW = Math.random() > 0.5;

      if (onC) {
        pg.position.set(bx * 50 + (Math.random() - 0.5) * 10, 0, bz * 50 + (Math.random() - 0.5) * 10);
        pg.rotation.y = Math.random() > 0.5 ? 0 : Math.PI / 2;
      } else {
        pg.position.set(
          isHW ? (Math.random() - 0.5) * 40 + bx * 50 : bx * 50 + side * 9,
          0,
          isHW ? bz * 50 + side * 9 : (Math.random() - 0.5) * 40 + bz * 50
        );
        pg.rotation.y = isHW ? (Math.random() > 0.5 ? 0 : Math.PI) : (Math.random() > 0.5 ? Math.PI / 2 : -Math.PI / 2);
      }

      scene.add(pg);
      const pData: PedestrianData = { mesh: pg, speed: 0.03 + Math.random() * 0.03, direction: pg.rotation.y, throwVelocity: new THREE.Vector3(), isThrown: false, blockX: bx, blockZ: bz, onCrossing: onC };
      pedestrians.push(pData);
      collisionObjects.push({ mesh: pg, type: 'pedestrian', radius: 0.3, data: pData });
    }

    // ── Cyclists ─────────────────────────────────────────────────────────────
    const cyclists: CyclistData[] = [];
    const sharedBikeBodyGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.8, 5);
    const sharedBikeHeadGeo = new THREE.SphereGeometry(0.2, 5, 5);
    const sharedBikeFrameGeo = new THREE.BoxGeometry(0.1, 0.1, 1);
    const sharedBikeWheelGeo = new THREE.TorusGeometry(0.3, 0.05, 4, 8);
    const bikeWheelMat = new THREE.MeshLambertMaterial({ color: 0x222222 });
    const bikeFrameMat = new THREE.MeshLambertMaterial({ color: 0x222222 });

    for (let i = 0; i < 12; i++) {
      const cg = new THREE.Group();
      const col = Math.floor(Math.random() * 0xffffff);
      const cMat = new THREE.MeshLambertMaterial({ color: col });

      const cb = new THREE.Mesh(sharedBikeBodyGeo, cMat); cb.position.y = 1.2; cb.rotation.z = 0.3; cg.add(cb);
      const ch = new THREE.Mesh(sharedBikeHeadGeo, skinMat); ch.position.set(0.1, 1.8, 0); cg.add(ch);
      const cf = new THREE.Mesh(sharedBikeFrameGeo, bikeFrameMat); cf.position.set(0, 0.5, 0); cg.add(cf);

      const fw = new THREE.Mesh(sharedBikeWheelGeo, bikeWheelMat); fw.rotation.y = Math.PI / 2; fw.position.set(0, 0.3, 0.5); cg.add(fw);
      const bw = new THREE.Mesh(sharedBikeWheelGeo, bikeWheelMat); bw.rotation.y = Math.PI / 2; bw.position.set(0, 0.3, -0.5); cg.add(bw);

      const lane = Math.floor(Math.random() * 10) - 5;
      const dir = Math.random() > 0.5 ? 1 : -1;
      const isH = Math.random() > 0.5;
      cg.position.set(
        isH ? (Math.random() - 0.5) * 480 : lane * 50 + 5,
        0,
        isH ? lane * 50 + 5 : (Math.random() - 0.5) * 480
      );
      cg.rotation.y = isH ? (dir > 0 ? 0 : Math.PI) : (dir > 0 ? Math.PI / 2 : -Math.PI / 2);

      scene.add(cg);
      const cData: CyclistData = { mesh: cg, speed: 0.15 + Math.random() * 0.15, direction: dir, isHorizontal: isH, throwVelocity: new THREE.Vector3(), isThrown: false, wheels: [fw, bw] };
      cyclists.push(cData);
      collisionObjects.push({ mesh: cg, type: 'cyclist', radius: 0.4, data: cData });
    }

    // ── Camera ───────────────────────────────────────────────────────────────
    camera.position.set(0, 8, -15);
    camera.lookAt(carGroup.position);

    // ── Input ────────────────────────────────────────────────────────────────
    const keys: Record<string, boolean> = {};
    let currentSpeed = 0;
    let currentRotation = 0;
    let currentGear = 'P';
    let lastGearPressTime = 0;

    const handleKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      keys[k] = true;
      if (k === 'p') { currentGear = 'P'; setGear('P'); }
      // D only switches to Drive gear when car is nearly stopped AND no other action key
      if (k === 'd' && Math.abs(currentSpeed) < 0.05) {
        const now = Date.now();
        if (now - lastGearPressTime > 300) { currentGear = 'D'; setGear('D'); lastGearPressTime = now; }
      }
      if (k === 'r' && Math.abs(currentSpeed) < 0.05) { currentGear = 'R'; setGear('R'); }
    };
    const handleKeyUp = (e: KeyboardEvent) => { keys[e.key.toLowerCase()] = false; };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    // ── Collision helpers ────────────────────────────────────────────────────
    const circleCircle = (x1: number, z1: number, r1: number, x2: number, z2: number, r2: number) => {
      const dx = x1 - x2, dz = z1 - z2;
      return (dx * dx + dz * dz) < (r1 + r2) * (r1 + r2);
    };
    const circleBox = (cx: number, cz: number, cr: number, bx: number, bz: number, bw: number, bd: number) => {
      const nx = Math.max(bx - bw / 2, Math.min(cx, bx + bw / 2));
      const nz = Math.max(bz - bd / 2, Math.min(cz, bz + bd / 2));
      const dx = cx - nx, dz = cz - nz;
      return (dx * dx + dz * dz) < cr * cr;
    };

    // ── Game state vars ───────────────────────────────────────────────────────
    let violationCount = 0;
    let scoreCount = 0;
    let damageCount = 0;
    let frame = 0;
    let isGameOver = false;
    let dayTime = 0; // advances each frame

    // ── Animation loop ────────────────────────────────────────────────────────
    const animate = () => {
      if (isGameOver) return;
      requestAnimationFrame(animate);
      frame++;

      // Day/night cycle (full cycle every ~20 min at 60fps = 72000 frames)
      dayTime = (dayTime + 1 / 72000) % 1;
      const sunAngle = dayTime * Math.PI * 2;
      const sunY = Math.sin(sunAngle);        // -1 (midnight) to 1 (noon)
      const sunX = Math.cos(sunAngle) * 200;
      sunLight.position.set(sunX, Math.max(sunY * 150, 5), 80);
      const dayFactor = Math.max(0, sunY);    // 0 at night, 1 at noon
      ambientLight.intensity = 0.2 + dayFactor * 0.6;
      sunLight.intensity = dayFactor;
      const skyR = 0x87 / 255, skyG = 0xCE / 255, skyB = 0xEB / 255;
      const nightR = 0x05 / 255, nightG = 0x05 / 255, nightB = 0x15 / 255;
      scene.background = new THREE.Color(
        skyR * dayFactor + nightR * (1 - dayFactor),
        skyG * dayFactor + nightG * (1 - dayFactor),
        skyB * dayFactor + nightB * (1 - dayFactor)
      );
      if (frame % 30 === 0) setTimeOfDay(dayTime);

      // ── Traffic light cycle ────────────────────────────────────────────────
      trafficLights.forEach(tl => {
        tl.timer++;
        const cycle = tl.timer % 396;
        if (cycle < 180) {
          tl.green.material.emissiveIntensity = 1;
          tl.yellow.material.emissiveIntensity = 0;
          tl.red.material.emissiveIntensity = 0;
          tl.state = 'green';
        } else if (cycle < 198) {
          tl.green.material.emissiveIntensity = 0;
          tl.yellow.material.emissiveIntensity = 1;
          tl.red.material.emissiveIntensity = 0;
          tl.state = 'yellow';
        } else {
          tl.green.material.emissiveIntensity = 0;
          tl.yellow.material.emissiveIntensity = 0;
          tl.red.material.emissiveIntensity = 1;
          tl.state = 'red';
        }
      });

      // ── Car physics ───────────────────────────────────────────────────────
      const maxSpeed = 2.5;
      const accel = 0.022;
      const decel = 0.025;
      const brakeF = 0.14;
      const turnSpeed = 0.032;

      if (keys['w']) {
        if (currentGear === 'D') currentSpeed = Math.min(currentSpeed + accel, maxSpeed);
        else if (currentGear === 'R') currentSpeed = Math.min(currentSpeed + decel * 2, 0);
      } else if (keys['s']) {
        if (currentGear === 'R') currentSpeed = Math.max(currentSpeed - accel * 0.7, -maxSpeed / 2.5);
        else if (currentGear === 'D') currentSpeed = Math.max(currentSpeed - decel * 2, 0);
      } else {
        if (currentGear === 'D') currentSpeed = Math.max(currentSpeed - decel * 0.5, 0);
        else if (currentGear === 'R') currentSpeed = Math.min(currentSpeed + decel * 0.5, 0);
        else currentSpeed *= 0.95;
      }

      if (keys[' ']) {
        if (currentSpeed > 0) currentSpeed = Math.max(currentSpeed - brakeF, 0);
        else if (currentSpeed < 0) currentSpeed = Math.min(currentSpeed + brakeF, 0);
      }

      // Turning
      if (keys['a']) currentRotation += turnSpeed * Math.min(Math.abs(currentSpeed / maxSpeed) * 2.5, 1);
      if (keys['d']) currentRotation -= turnSpeed * Math.min(Math.abs(currentSpeed / maxSpeed) * 2.5, 1);

      // Intended position
      const ix = carGroup.position.x + Math.sin(currentRotation) * currentSpeed;
      const iz = carGroup.position.z + Math.cos(currentRotation) * currentSpeed;

      // World bounds (soft wrap)
      const WORLD = 290;
      const boundedX = Math.max(-WORLD, Math.min(WORLD, ix));
      const boundedZ = Math.max(-WORLD, Math.min(WORLD, iz));

      let canMove = true;
      let impactThisFrame = false;

      // ── Collision checks ──────────────────────────────────────────────────
      for (const obj of collisionObjects) {
        // Manhattan early exit — skip far objects cheaply
        const mdx = Math.abs(boundedX - obj.mesh.position.x);
        const mdz = Math.abs(boundedZ - obj.mesh.position.z);
        if (mdx + mdz > 30) continue;

        let hit = false;
        if (obj.isBox) {
          hit = circleBox(boundedX, boundedZ, 1.6, obj.mesh.position.x, obj.mesh.position.z, obj.width!, obj.depth!);
        } else {
          hit = circleCircle(boundedX, boundedZ, 1.6, obj.mesh.position.x, obj.mesh.position.z, obj.radius!);
        }

        if (!hit) continue;

        const impact = Math.abs(currentSpeed);

        if (obj.type === 'building') {
          canMove = false;
          const dmg = impact * 12;
          if (dmg > 0.5 && !impactThisFrame) { audio.playImpact(impact); impactThisFrame = true; }
          damageCount = Math.min(100, damageCount + dmg);
          currentSpeed *= 0.04;
          setDamage(Math.floor(damageCount));
        } else if (obj.type === 'lamp' || obj.type === 'pole') {
          canMove = false;
          const dmg = impact * 8;
          if (dmg > 0.5 && !impactThisFrame) { audio.playImpact(impact * 0.6); impactThisFrame = true; }
          damageCount = Math.min(100, damageCount + dmg);
          currentSpeed *= 0.08;
          setDamage(Math.floor(damageCount));
        } else if (obj.type === 'vehicle') {
          canMove = false;
          const dmg = impact * 8;
          if (dmg > 0.5 && !impactThisFrame) { audio.playImpact(impact * 0.8); impactThisFrame = true; }
          damageCount = Math.min(100, damageCount + dmg);
          currentSpeed *= 0.18;
          violationCount++;
          setViolations(violationCount);
          setDamage(Math.floor(damageCount));
        } else if (obj.type === 'pedestrian') {
          const pData = obj.data as PedestrianData;
          if (!pData.isThrown && impact > 0.08) {
            pData.isThrown = true;
            const dir = new THREE.Vector3(Math.sin(currentRotation), 0.18, Math.cos(currentRotation));
            pData.throwVelocity.copy(dir.multiplyScalar(impact * 4.5));
            violationCount += 10;
            damageCount = Math.min(100, damageCount + 20);
            audio.playImpact(0.9);
            setViolations(violationCount);
            setDamage(Math.floor(damageCount));
          }
        } else if (obj.type === 'cyclist') {
          const cData = obj.data as CyclistData;
          if (!cData.isThrown && impact > 0.08) {
            cData.isThrown = true;
            const dir = new THREE.Vector3(Math.sin(currentRotation), 0.14, Math.cos(currentRotation));
            cData.throwVelocity.copy(dir.multiplyScalar(impact * 3.5));
            violationCount += 5;
            damageCount = Math.min(100, damageCount + 15);
            audio.playImpact(0.7);
            setViolations(violationCount);
            setDamage(Math.floor(damageCount));
          }
        }
      }

      if (damageCount >= 100) {
        isGameOver = true;
        setGameOver(true);
      }

      // Move car
      if (canMove) {
        carGroup.position.x = boundedX;
        carGroup.position.z = boundedZ;
      }
      carGroup.rotation.y = currentRotation;

      // Wheel spin
      wheels.forEach(w => { w.rotation.x += currentSpeed * 0.5; });

      // Camera follow (third-person)
      const camOff = new THREE.Vector3(0, 8, -15);
      camOff.applyAxisAngle(new THREE.Vector3(0, 1, 0), currentRotation);
      camera.position.x = carGroup.position.x + camOff.x;
      camera.position.y = carGroup.position.y + camOff.y;
      camera.position.z = carGroup.position.z + camOff.z;
      camera.lookAt(carGroup.position);

      // ── AI vehicles (EVERY frame for smooth movement) ──────────────────────
      for (const v of otherVehicles) {
        // Traffic light stopping
        if (v.stoppedAtLight && v.stopTimer > 0) { v.stopTimer--; continue; }

        let stop = false;
        for (const tl of trafficLights) {
          if (tl.state === 'green') continue;
          const mDist = Math.abs(v.mesh.position.x - tl.position.x) + Math.abs(v.mesh.position.z - tl.position.z);
          if (mDist > 50) continue;
          const dist = v.mesh.position.distanceTo(tl.position);
          if (dist < 18 && dist > 4) {
            const approaching = v.isHorizontal
              ? (v.direction > 0 && v.mesh.position.x < tl.position.x) || (v.direction < 0 && v.mesh.position.x > tl.position.x)
              : (v.direction > 0 && v.mesh.position.z < tl.position.z) || (v.direction < 0 && v.mesh.position.z > tl.position.z);
            if (approaching) { stop = true; break; }
          }
        }

        if (stop) { v.stoppedAtLight = true; v.stopTimer = 80 + Math.floor(Math.random() * 40); continue; }
        v.stoppedAtLight = false;

        // Check for vehicle ahead (avoid rear-ending)
        let tooClose = false;
        const lookAhead = 8;
        for (const other of otherVehicles) {
          if (other === v) continue;
          const dx = other.mesh.position.x - v.mesh.position.x;
          const dz = other.mesh.position.z - v.mesh.position.z;
          if (v.isHorizontal) {
            if (Math.abs(dz) < 4 && dx * v.direction > 0 && Math.abs(dx) < lookAhead) { tooClose = true; break; }
          } else {
            if (Math.abs(dx) < 4 && dz * v.direction > 0 && Math.abs(dz) < lookAhead) { tooClose = true; break; }
          }
        }
        if (tooClose) continue;

        const nx = v.isHorizontal ? v.mesh.position.x + v.speed * v.direction : v.mesh.position.x;
        const nz = v.isHorizontal ? v.mesh.position.z : v.mesh.position.z + v.speed * v.direction;

        // Check static obstacle collision
        let ok = true;
        if (frame % 3 === 0) {
          for (const s of collisionObjects) {
            if (s.type !== 'building' && s.type !== 'lamp' && s.type !== 'pole') continue;
            const rough = Math.abs(nx - s.mesh.position.x) + Math.abs(nz - s.mesh.position.z);
            if (rough > 20) continue;
            const hit = s.isBox
              ? circleBox(nx, nz, 3, s.mesh.position.x, s.mesh.position.z, s.width!, s.depth!)
              : circleCircle(nx, nz, 3, s.mesh.position.x, s.mesh.position.z, s.radius!);
            if (hit) { ok = false; break; }
          }
        }

        if (ok) {
          v.mesh.position.x = nx;
          v.mesh.position.z = nz;
          if (v.isHorizontal && Math.abs(nx) > 280) v.mesh.position.x = -280 * v.direction;
          if (!v.isHorizontal && Math.abs(nz) > 280) v.mesh.position.z = -280 * v.direction;
        }
        v.wheels.forEach(w => { w.rotation.x += v.speed * 0.3; });
      }

      // ── Pedestrians (every 2 frames for smooth walk) ───────────────────────
      if (frame % 2 === 0) {
        for (const p of pedestrians) {
          if (p.isThrown) {
            p.mesh.position.add(p.throwVelocity);
            p.throwVelocity.y -= 0.015;
            p.mesh.rotation.x += 0.08;
            if (p.mesh.position.y <= 0) { p.mesh.position.y = 0; p.isThrown = false; p.throwVelocity.set(0, 0, 0); p.mesh.rotation.x = 0; }
            continue;
          }
          // Walk with arm swing animation
          const walkPhase = frame * 0.05 + p.speed * 100;
          p.mesh.children[2].rotation.x = Math.sin(walkPhase) * 0.4; // left arm
          p.mesh.children[3].rotation.x = -Math.sin(walkPhase) * 0.4; // right arm

          p.mesh.position.x += Math.cos(p.direction) * p.speed;
          p.mesh.position.z += Math.sin(p.direction) * p.speed;
          const bx = p.blockX * 50, bz = p.blockZ * 50;
          const limit = p.onCrossing ? 6 : 25;
          if (Math.abs(p.mesh.position.x - bx) > limit) { p.direction = Math.PI - p.direction; p.mesh.rotation.y = p.direction; }
          if (Math.abs(p.mesh.position.z - bz) > limit) { p.direction = -p.direction; p.mesh.rotation.y = p.direction; }
          if (Math.random() < 0.003) { p.direction += Math.random() > 0.5 ? Math.PI / 2 : -Math.PI / 2; p.mesh.rotation.y = p.direction; }
        }
      }

      // ── Cyclists (every 2 frames) ──────────────────────────────────────────
      if (frame % 2 === 0) {
        for (const b of cyclists) {
          if (b.isThrown) {
            b.mesh.position.add(b.throwVelocity);
            b.throwVelocity.y -= 0.015;
            b.mesh.rotation.x += 0.12;
            if (b.mesh.position.y <= 0) { b.mesh.position.y = 0; b.isThrown = false; b.throwVelocity.set(0, 0, 0); b.mesh.rotation.x = 0; }
            continue;
          }
          if (b.isHorizontal) {
            b.mesh.position.x += b.speed * b.direction;
            if (Math.abs(b.mesh.position.x) > 280) b.mesh.position.x = -280 * b.direction;
          } else {
            b.mesh.position.z += b.speed * b.direction;
            if (Math.abs(b.mesh.position.z) > 280) b.mesh.position.z = -280 * b.direction;
          }
          b.wheels.forEach(w => { w.rotation.y += b.speed * 0.5; });
        }
      }

      // ── HUD updates ───────────────────────────────────────────────────────
      if (frame % 6 === 0) {
        setSpeed(Math.round(Math.abs(currentSpeed) * 100));
        setRpm(Math.round(Math.abs(currentSpeed / maxSpeed) * 7000 + 800));
        setCarPos({ x: Math.round(carGroup.position.x), z: Math.round(carGroup.position.z) });
      }

      if (frame % 60 === 0 && currentSpeed > 0.08 && damageCount < 100) {
        scoreCount++;
        setScore(scoreCount);
      }

      // ── Animate clouds & trees ──────────────────────────────────────────
      for (const c of clouds) {
        (c as any).position.x += (c as any).cloudSpeed || 0.03;
        if ((c as any).position.x > 350) (c as any).position.x = -350;
      }
      const swayT = frame * 0.02;
      for (const t of swayTrees) {
        t.rotation.z = Math.sin(swayT + t.position.x * 0.1) * 0.02;
        t.rotation.x = Math.cos(swayT * 0.7 + t.position.z * 0.1) * 0.015;
      }

      // Engine audio
      audio.setEngine(currentSpeed, maxSpeed);

      renderer.render(scene, camera);
    };

    animate();

    // ── Resize ────────────────────────────────────────────────────────────────
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', onResize);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('resize', onResize);
      if (mountRef.current?.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    if (gameInitialized.current) return;
    gameInitialized.current = true;
    const cleanup = initGame();
    return () => { cleanup?.(); };
  }, [initGame]);

  // ── Restart (no page reload) ───────────────────────────────────────────────
  const handleRestart = () => {
    // Tear down old scene then re-init
    setGameOver(false);
    setSpeed(0);
    setGear('P');
    setScore(0);
    setViolations(0);
    setDamage(0);
    setCarPos({ x: 0, z: 0 });
    gameInitialized.current = false;
    // Clear canvas
    if (mountRef.current) {
      while (mountRef.current.firstChild) mountRef.current.removeChild(mountRef.current.firstChild);
    }
    setTimeout(() => {
      gameInitialized.current = false;
      const cleanup = initGame();
      return () => { cleanup?.(); };
    }, 50);
  };

  // ── Minimap ────────────────────────────────────────────────────────────────
  const minimapSize = 160;
  const worldSize = 580;
  const minimapScale = minimapSize / worldSize;
  const dotX = (carPos.x / worldSize + 0.5) * minimapSize;
  const dotZ = (carPos.z / worldSize + 0.5) * minimapSize;

  // ── Speedometer arc ───────────────────────────────────────────────────────
  const speedPct = Math.min(speed / 120, 1);
  const arcStart = Math.PI * 0.75;
  const arcEnd = arcStart + speedPct * Math.PI * 1.5;
  const arcR = 38;
  const arcCX = 50, arcCY = 55;
  const arcPath = (start: number, end: number, r: number) => {
    const sx = arcCX + r * Math.cos(start);
    const sy = arcCY + r * Math.sin(start);
    const ex = arcCX + r * Math.cos(end);
    const ey = arcCY + r * Math.sin(end);
    const large = (end - start) > Math.PI ? 1 : 0;
    return `M ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`;
  };

  // Damage color
  const dmgColor = damage > 70 ? '#ff3333' : damage > 40 ? '#ff9900' : '#00ff88';
  const isDaytime = Math.sin(timeOfDay * Math.PI * 2) > 0;
  const timeLabel = isDaytime ? '☀️' : '🌙';

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden', position: 'relative', fontFamily: "'Courier New', monospace" }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} onClick={() => audioRef.current.resume()} />

      {/* ── Main HUD ── */}
      <div style={{
        position: 'absolute', top: 16, left: 16,
        background: 'linear-gradient(135deg, rgba(0,0,0,0.85) 0%, rgba(10,20,10,0.9) 100%)',
        border: '1px solid #00ff88',
        borderRadius: 12, padding: '14px 18px',
        boxShadow: '0 0 20px rgba(0,255,136,0.2), inset 0 0 30px rgba(0,0,0,0.5)',
        minWidth: 170
      }}>
        {/* Speedometer */}
        <svg width="100" height="78" style={{ display: 'block', margin: '0 auto 8px' }}>
          {/* Background arc */}
          <path d={arcPath(arcStart, arcStart + Math.PI * 1.5, arcR)} fill="none" stroke="#1a3320" strokeWidth={8} strokeLinecap="round" />
          {/* Speed arc */}
          {speed > 0 && <path d={arcPath(arcStart, arcEnd, arcR)} fill="none"
            stroke={speedPct > 0.75 ? '#ff3333' : speedPct > 0.5 ? '#ffaa00' : '#00ff88'}
            strokeWidth={8} strokeLinecap="round" />}
          {/* Speed text */}
          <text x={arcCX} y={arcCY - 4} textAnchor="middle" fill="#ffffff" fontSize={18} fontWeight="bold" fontFamily="monospace">{speed}</text>
          <text x={arcCX} y={arcCY + 12} textAnchor="middle" fill="#888" fontSize={9} fontFamily="monospace">km/h</text>
        </svg>

        <div style={{ borderTop: '1px solid #1a3a1a', paddingTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#666', fontSize: 11 }}>GEAR</span>
            <span style={{
              color: gear === 'R' ? '#ff9900' : gear === 'P' ? '#888' : '#00ff88',
              fontSize: 20, fontWeight: 'bold', letterSpacing: 2
            }}>{gear}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#666', fontSize: 11 }}>RPM</span>
            <span style={{ color: '#aaffcc', fontSize: 11 }}>{rpm.toLocaleString()}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#666', fontSize: 11 }}>SCORE</span>
            <span style={{ color: '#ffffff', fontSize: 13, fontWeight: 'bold' }}>{score}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#666', fontSize: 11 }}>VIOLATIONS</span>
            <span style={{ color: violations > 5 ? '#ff3333' : violations > 0 ? '#ffaa00' : '#00ff88', fontSize: 13, fontWeight: 'bold' }}>{violations}</span>
          </div>
          {/* Damage bar */}
          <div style={{ marginTop: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ color: '#666', fontSize: 10 }}>DAMAGE</span>
              <span style={{ color: dmgColor, fontSize: 10, fontWeight: 'bold' }}>{damage}%</span>
            </div>
            <div style={{ background: '#111', borderRadius: 3, height: 6, overflow: 'hidden', border: '1px solid #1a3a1a' }}>
              <div style={{ width: `${damage}%`, height: '100%', background: `linear-gradient(90deg, #00ff88, ${dmgColor})`, transition: 'width 0.3s, background 0.3s', borderRadius: 3 }} />
            </div>
          </div>
          {/* Time of day */}
          <div style={{ textAlign: 'center', fontSize: 11, color: '#555', marginTop: 2 }}>
            {timeLabel} {isDaytime ? 'Daytime' : 'Nighttime'}
          </div>
        </div>
      </div>

      {/* ── Minimap ── */}
      <div style={{
        position: 'absolute', bottom: 16, right: 16,
        background: 'rgba(0,0,0,0.82)',
        border: '1px solid #00ff88',
        borderRadius: 10,
        padding: 6,
        boxShadow: '0 0 16px rgba(0,255,136,0.15)'
      }}>
        <div style={{ color: '#444', fontSize: 9, textAlign: 'center', marginBottom: 3, letterSpacing: 1 }}>MINIMAP</div>
        <div style={{ position: 'relative', width: minimapSize, height: minimapSize, background: '#0a120a', borderRadius: 6, overflow: 'hidden' }}>
          {/* Road grid */}
          {[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map(i => (
            <React.Fragment key={i}>
              <div style={{ position: 'absolute', left: 0, right: 0, height: 2, background: '#1a2a1a', top: ((i * 50 + 290) / worldSize) * minimapSize - 1 }} />
              <div style={{ position: 'absolute', top: 0, bottom: 0, width: 2, background: '#1a2a1a', left: ((i * 50 + 290) / worldSize) * minimapSize - 1 }} />
            </React.Fragment>
          ))}
          {/* Car dot */}
          <div style={{
            position: 'absolute',
            width: 7, height: 7,
            background: '#ff3333',
            borderRadius: '50%',
            boxShadow: '0 0 6px #ff3333',
            left: dotX - 3.5,
            top: dotZ - 3.5,
            transform: 'translate(0,0)'
          }} />
          {/* Compass */}
          <div style={{ position: 'absolute', top: 4, right: 5, color: '#335533', fontSize: 8 }}>N</div>
        </div>
        <div style={{ color: '#336633', fontSize: 8, textAlign: 'center', marginTop: 3 }}>{carPos.x}, {carPos.z}</div>
      </div>

      {/* ── Controls ── */}
      <div style={{
        position: 'absolute', bottom: 16, left: 16,
        background: 'rgba(0,0,0,0.75)',
        border: '1px solid #1a3a1a',
        borderRadius: 10, padding: '12px 14px',
        color: '#555', fontSize: 11, lineHeight: 1.7
      }}>
        <div style={{ color: '#336633', fontWeight: 'bold', marginBottom: 4, fontSize: 12 }}>CONTROLS</div>
        {[['W', 'Accelerate'], ['S', 'Brake/Reverse'], ['A / D', 'Steer'], ['SPACE', 'Emergency Brake'], ['P / R', 'Park / Reverse gear']].map(([k, v]) => (
          <div key={k} style={{ display: 'flex', gap: 8 }}>
            <span style={{ color: '#00aa55', minWidth: 60 }}>{k}</span>
            <span>{v}</span>
          </div>
        ))}
        <div style={{ marginTop: 6, borderTop: '1px solid #1a2a1a', paddingTop: 6, color: '#335533', fontSize: 10 }}>
          Drive slow for Drive gear (D key)
        </div>
      </div>

      {/* ── Feature list ── */}
      <div style={{
        position: 'absolute', top: 16, right: 16,
        background: 'rgba(0,0,0,0.75)',
        border: '1px solid #1a3a1a',
        borderRadius: 10, padding: '12px 14px',
        color: '#444', fontSize: 11, lineHeight: 1.8, maxWidth: 230
      }}>
        <div style={{ color: '#336633', fontWeight: 'bold', marginBottom: 4, fontSize: 12 }}>FEATURES</div>
        {[
          'Building AABB collision',
          'Pedestrians thrown on impact',
          'Traffic light system (30s)',
          'AI vehicles obey signals',
          'Day / night cycle',
          'Minimap + coordinates',
          'Engine audio feedback',
          'Realistic RPM & damage bar',
          'World bounds enforced',
          'Game over at 100% damage'
        ].map(f => <div key={f} style={{ color: '#2a6a2a' }}>✓ <span style={{ color: '#445544' }}>{f}</span></div>)}
      </div>

      {/* ── Critical damage warning ── */}
      {damage > 80 && !gameOver && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#ff3333', fontSize: 40, fontWeight: 900,
          background: 'rgba(0,0,0,0.92)',
          padding: '20px 50px', borderRadius: 16,
          border: '3px solid #ff3333',
          textAlign: 'center',
          animation: 'pulse 0.6s infinite alternate',
          pointerEvents: 'none'
        }}>
          ⚠ CRITICAL DAMAGE ⚠
          <div style={{ fontSize: 16, color: '#ff8800', marginTop: 6, fontWeight: 400 }}>AVOID COLLISIONS</div>
        </div>
      )}

      {/* ── Game Over ── */}
      {gameOver && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(0,0,0,0.96)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{ color: '#ff2222', fontSize: 64, fontWeight: 900, letterSpacing: 6, textShadow: '0 0 40px #ff0000', marginBottom: 8 }}>
            GAME OVER
          </div>
          <div style={{ color: '#335533', fontSize: 13, marginBottom: 30, letterSpacing: 3 }}>VEHICLE DESTROYED</div>
          <div style={{
            background: 'rgba(0,20,0,0.8)', border: '1px solid #1a4a1a', borderRadius: 12,
            padding: '20px 40px', marginBottom: 36, display: 'flex', flexDirection: 'column', gap: 12, minWidth: 280
          }}>
            {[['FINAL SCORE', score, '#ffffff'], ['VIOLATIONS', violations, violations > 5 ? '#ff6666' : '#ffaa44'], ['DAMAGE', `${damage}%`, '#ff4444']].map(([label, val, color]) => (
              <div key={label as string} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#446644', fontSize: 12, letterSpacing: 2 }}>{label}</span>
                <span style={{ color: color as string, fontSize: 22, fontWeight: 'bold' }}>{val}</span>
              </div>
            ))}
          </div>
          <button onClick={handleRestart} style={{
            padding: '14px 48px', fontSize: 18, fontWeight: 900,
            background: 'linear-gradient(135deg, #00cc55, #009944)',
            color: '#000', border: 'none', borderRadius: 10, cursor: 'pointer',
            letterSpacing: 3, boxShadow: '0 0 30px rgba(0,255,100,0.4)',
            transition: 'all 0.2s'
          }}
            onMouseEnter={e => { (e.target as HTMLButtonElement).style.transform = 'scale(1.06)'; }}
            onMouseLeave={e => { (e.target as HTMLButtonElement).style.transform = 'scale(1)'; }}
          >
            RESTART
          </button>
        </div>
      )}

      {/* ── Pulse keyframe ── */}
      <style>{`@keyframes pulse { from { opacity:1; } to { opacity:0.3; } }`}</style>
    </div>
  );
};

export default OpenWorldDrivingSim;
