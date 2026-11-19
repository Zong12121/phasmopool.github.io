import * as CANNON from 'cannon-es';
import * as THREE from 'three';
import { NeuralNetwork } from '../ai/NeuralNetwork';
import { COLLISION_GROUP_ROBOT, COLLISION_GROUP_WORLD } from './BipedWalker3D';
import type { ProceduralAssets } from './materials';
import type { SimpleModeAgent } from './OtherModes';
import type { ActivationFn, AgentSnapshot } from './types';

export const MANUAL_CAR_INPUT_LABELS = [
  'Луч -50°',
  'Луч -25°',
  'Луч 0°',
  'Луч +25°',
  'Луч +50°',
  'Смещение от центра',
  'Курс sin(Δψ)',
  'Курс cos(Δψ)',
  'Поворот 8 м',
  'Поворот 22 м',
  'Скорость',
  'Боковой снос',
  'Обороты (RPM)',
  'Передача (N/1–5)',
  'Педаль сцепления',
  'Подсказка тахометра',
];

export const MANUAL_CAR_OUTPUT_LABELS = [
  'Руль',
  'Газ',
  'Тормоз',
  'Сцепление',
  'Рычаг КПП',
];

export interface TrackSample {
  x: number;
  z: number;
  tx: number; // unit tangent X
  tz: number; // unit tangent Z
  nx: number; // unit right normal X
  nz: number; // unit right normal Z
  curvature: number; // signed rad/m (+ = right turn, - = left turn)
  s: number; // arc length from start (m)
}

export interface CircuitData {
  curve: THREE.CatmullRomCurve3;
  samples: TrackSample[];
  totalLength: number;
  halfWidth: number;
}

const GEAR_RATIOS: Record<number, number> = {
  [-1]: -3.1,
  0: 0,
  1: 3.65,
  2: 2.25,
  3: 1.5,
  4: 1.08,
  5: 0.82,
};

export function createManualCarCircuitData(): CircuitData {
  // Smooth closed sweeping test track (~240m) starting at (0, 0, 0) heading +Z
  const waypoints = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, 26),
    new THREE.Vector3(8, 0, 50),
    new THREE.Vector3(28, 0, 62),
    new THREE.Vector3(48, 0, 54),
    new THREE.Vector3(56, 0, 32),
    new THREE.Vector3(46, 0, 12),
    new THREE.Vector3(54, 0, -12),
    new THREE.Vector3(44, 0, -34),
    new THREE.Vector3(18, 0, -40),
    new THREE.Vector3(2, 0, -24),
  ];
  const curve = new THREE.CatmullRomCurve3(waypoints, true, 'catmullrom', 0.45);
  const totalLength = curve.getLength();
  const numSamples = 320;
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < numSamples; i++) {
    pts.push(curve.getPointAt(i / numSamples));
  }

  const samples: TrackSample[] = [];
  let accumS = 0;
  for (let i = 0; i < numSamples; i++) {
    const prev = pts[(i - 1 + numSamples) % numSamples];
    const curr = pts[i];
    const next = pts[(i + 1) % numSamples];

    if (i > 0) {
      accumS += Math.hypot(curr.x - prev.x, curr.z - prev.z);
    }

    const dx = next.x - prev.x;
    const dz = next.z - prev.z;
    const len = Math.hypot(dx, dz) || 1;
    const tx = dx / len;
    const tz = dz / len;
    // Right-hand perpendicular in XZ
    const nx = -tz;
    const nz = tx;

    const dx1 = curr.x - prev.x;
    const dz1 = curr.z - prev.z;
    const dx2 = next.x - curr.x;
    const dz2 = next.z - curr.z;
    const a1 = Math.atan2(dx1, dz1);
    const a2 = Math.atan2(dx2, dz2);
    let dPsi = a2 - a1;
    while (dPsi > Math.PI) dPsi -= Math.PI * 2;
    while (dPsi < -Math.PI) dPsi += Math.PI * 2;
    const ds = Math.hypot(dx2, dz2) || 1;
    const curvature = THREE.MathUtils.clamp(dPsi / ds, -0.25, 0.25);

    samples.push({
      x: curr.x,
      z: curr.z,
      tx,
      tz,
      nx,
      nz,
      curvature,
      s: accumS,
    });
  }

  return {
    curve,
    samples,
    totalLength,
    halfWidth: 4.1,
  };
}

export class ManualCarAgent3D implements SimpleModeAgent {
  public readonly id: number;
  public readonly brain: NeuralNetwork;
  public alive = true;
  public fitness = 0;
  public distance = 0;
  public maxDistance = 0;
  public readonly sensorBuffer = new Float32Array(16);
  public readonly rootGroup = new THREE.Group();

  private readonly body: CANNON.Body;
  private readonly world: CANNON.World;
  private readonly circuit: CircuitData;

  private age = 0;
  private heading = 0; // rad, 0 = +Z
  private trackIdx = 0;
  private lastS = 0;
  private totalProgress = 0;

  // Manual Transmission & Engine State
  private gear = 0; // Starts in Neutral -> AI must clutch & shift to 1st!
  private engineRpm = 950;
  private clutchPedal = 1.0; // 1 = pressed (disengaged), 0 = engaged
  private throttleInput = 0;
  private brakeInput = 0;
  private shiftCooldown = 0;
  private lowProgressTimer = 0;
  private stallTimer = 0;
  private shiftBonus = 0;
  private stalled = false;
  private disqualifiedReason = '';
  private ghosted = false;

  private wheelSpin = 0;
  private currentSteer = 0;
  private readonly mats: THREE.MeshStandardMaterial[] = [];
  private coreMat!: THREE.MeshStandardMaterial;
  private brakeLightMat!: THREE.MeshStandardMaterial;
  private meshGroup: THREE.Group;
  private frontWheelPivots: THREE.Group[] = [];
  private wheelMeshes: THREE.Mesh[] = [];

  constructor(
    id: number,
    brain: NeuralNetwork,
    world: CANNON.World,
    physicsMat: CANNON.Material,
    assets: ProceduralAssets | null,
    circuit: CircuitData,
    headless = false,
  ) {
    this.id = id;
    this.brain = brain;
    this.world = world;
    this.circuit = circuit;

    const startSample = circuit.samples[0];
    this.heading = Math.atan2(startSample.tx, startSample.tz);

    this.body = new CANNON.Body({
      mass: 24,
      position: new CANNON.Vec3(startSample.x, 0.28, startSample.z),
      material: physicsMat,
      linearDamping: 0.04,
      angularDamping: 0.85,
      collisionFilterGroup: 0,
      collisionFilterMask: 0,
    });
    this.body.addShape(new CANNON.Box(new CANNON.Vec3(0.55, 0.22, 1.05)));
    this.body.fixedRotation = true;
    this.body.updateMassProperties();
    this.body.quaternion.setFromEuler(0, this.heading, 0);
    this.body.allowSleep = false;
    world.addBody(this.body);

    this.meshGroup = new THREE.Group();
    if (!headless && assets) {
      this.buildCarVisuals(assets);
      this.syncVisuals();
    }
  }

  private buildCarVisuals(assets: ProceduralAssets): void {
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      map: assets.brushedMetalTexture,
      roughness: 0.26,
      metalness: 0.78,
    });
    const carbonMat = new THREE.MeshStandardMaterial({
      color: 0x171e29,
      map: assets.carbonTexture,
      roughness: 0.62,
      metalness: 0.28,
    });
    const tireMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.9,
      metalness: 0.1,
    });
    const rimMat = new THREE.MeshStandardMaterial({
      color: 0xcbd5e1,
      roughness: 0.22,
      metalness: 0.9,
    });
    this.coreMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 1.4,
      roughness: 0.2,
    });
    this.brakeLightMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0x991b1b,
      emissiveIntensity: 0.6,
      roughness: 0.3,
    });
    this.mats.push(bodyMat, carbonMat, tireMat, rimMat, this.coreMat, this.brakeLightMat);

    // Lower sports chassis
    const lowerHull = new THREE.Mesh(new THREE.BoxGeometry(1.04, 0.24, 2.08), bodyMat);
    lowerHull.position.set(0, 0.02, 0);
    lowerHull.castShadow = true;
    this.meshGroup.add(lowerHull);

    // Cockpit cabin canopy
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.22, 1.02), carbonMat);
    cabin.position.set(0, 0.23, -0.08);
    cabin.castShadow = true;
    this.meshGroup.add(cabin);

    // Roof LiDAR / Telemetry pod
    const roofPod = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.14, 0.08, 12), this.coreMat);
    roofPod.position.set(0, 0.37, 0.05);
    this.meshGroup.add(roofPod);

    // Rear spoiler wing
    const wing = new THREE.Mesh(new THREE.BoxGeometry(1.02, 0.04, 0.22), carbonMat);
    wing.position.set(0, 0.28, -0.96);
    this.meshGroup.add(wing);

    // Headlights & Taillights
    for (const sx of [-0.36, 0.36]) {
      const head = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.07, 0.05), this.coreMat);
      head.position.set(sx, 0.05, 1.04);
      this.meshGroup.add(head);

      const tail = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.07, 0.05), this.brakeLightMat);
      tail.position.set(sx, 0.06, -1.04);
      this.meshGroup.add(tail);
    }

    // 4 wheels (front 2 inside steer pivots)
    const wheelGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.16, 16);
    wheelGeo.rotateZ(Math.PI / 2);
    const hubGeo = new THREE.CylinderGeometry(0.13, 0.13, 0.17, 10);
    hubGeo.rotateZ(Math.PI / 2);

    const wheelCoords: Array<[number, number, boolean]> = [
      [-0.55, 0.66, true],
      [0.55, 0.66, true],
      [-0.55, -0.66, false],
      [0.55, -0.66, false],
    ];
    for (const [wx, wz, isFront] of wheelCoords) {
      const pivot = new THREE.Group();
      pivot.position.set(wx, -0.1, wz);

      const wMesh = new THREE.Mesh(wheelGeo, tireMat);
      wMesh.castShadow = true;
      const hub = new THREE.Mesh(hubGeo, rimMat);
      wMesh.add(hub);

      pivot.add(wMesh);
      this.meshGroup.add(pivot);
      this.wheelMeshes.push(wMesh);
      if (isFront) this.frontWheelPivots.push(pivot);
    }

    this.rootGroup.add(this.meshGroup);
  }

  public getTrackPosition(): THREE.Vector3 {
    return new THREE.Vector3(this.body.position.x, this.body.position.y + 0.3, this.body.position.z);
  }

  public setHighlighted(v: boolean): void {
    if (this.ghosted || !this.coreMat) return;
    this.coreMat.color.setHex(v ? 0xf59e0b : 0x38bdf8);
    this.coreMat.emissive.setHex(v ? 0xd97706 : 0x0284c7);
    this.coreMat.emissiveIntensity = v ? 2.2 : 1.1;
  }

  private updateNearestSample(): { sample: TrackSample; lateral: number } {
    const samples = this.circuit.samples;
    const n = samples.length;
    const px = this.body.position.x;
    const pz = this.body.position.z;

    let bestIdx = this.trackIdx;
    let bestDistSq = Infinity;
    for (let offset = -18; offset <= 24; offset++) {
      const idx = (this.trackIdx + offset + n) % n;
      const sp = samples[idx];
      const dx = px - sp.x;
      const dz = pz - sp.z;
      const d2 = dx * dx + dz * dz;
      if (d2 < bestDistSq) {
        bestDistSq = d2;
        bestIdx = idx;
      }
    }

    this.trackIdx = bestIdx;
    const sp = samples[bestIdx];
    const lateral = (px - sp.x) * sp.nx + (pz - sp.z) * sp.nz;
    return { sample: sp, lateral };
  }

  private castTrackRay(angleOffset: number, lateral: number, headingErr: number, fwdCurv: number): number {
    // Ray distance to road edges considering current lateral offset and ray heading
    const rayAngle = headingErr + angleOffset;
    const cosA = Math.cos(rayAngle);
    const sinA = Math.sin(rayAngle);
    const maxRay = 18.0;
    const halfW = this.circuit.halfWidth;

    let d = maxRay;
    // Lateral intersection with left (-halfW) or right (+halfW) track border (adjusted slightly by curvature)
    const effectiveRate = sinA - 0.5 * fwdCurv * cosA * 3.5;
    if (Math.abs(effectiveRate) > 0.02) {
      const targetWall = effectiveRate > 0 ? halfW : -halfW;
      const hitDist = (targetWall - lateral) / effectiveRate;
      if (hitDist > 0.1 && hitDist < d) d = hitDist;
    }
    if (cosA < -0.25) {
      d = Math.min(d, 3.0);
    }
    return THREE.MathUtils.clamp(d / maxRay, 0, 1);
  }

  public stepControl(dt: number): void {
    if (!this.alive) return;
    this.age += dt;
    this.shiftCooldown = Math.max(0, this.shiftCooldown - dt);
    this.body.position.y = 0.28;
    this.body.velocity.y = 0;

    const { sample, lateral } = this.updateNearestSample();

    // Update signed forward progress along closed circuit
    let ds = sample.s - this.lastS;
    if (ds < -this.circuit.totalLength * 0.5) ds += this.circuit.totalLength;
    if (ds > this.circuit.totalLength * 0.5) ds -= this.circuit.totalLength;
    this.lastS = sample.s;
    this.totalProgress += ds;
    this.distance = Math.max(0, this.totalProgress);
    if (this.distance > this.maxDistance) this.maxDistance = this.distance;

    // Off-track curb disqualification
    if (Math.abs(lateral) > this.circuit.halfWidth + 0.7) {
      this.disqualify('Выезд с трассы');
      return;
    }

    const fwdX = Math.sin(this.heading);
    const fwdZ = Math.cos(this.heading);
    const rightX = -fwdZ;
    const rightZ = fwdX;

    const vFwd = this.body.velocity.x * fwdX + this.body.velocity.z * fwdZ;
    const vLat = this.body.velocity.x * rightX + this.body.velocity.z * rightZ;

    const trackHeading = Math.atan2(sample.tx, sample.tz);
    let headingErr = this.heading - trackHeading;
    while (headingErr > Math.PI) headingErr -= Math.PI * 2;
    while (headingErr < -Math.PI) headingErr += Math.PI * 2;

    const samples = this.circuit.samples;
    const curvNear = samples[(this.trackIdx + 10) % samples.length].curvature;
    const curvFar = samples[(this.trackIdx + 26) % samples.length].curvature;

    const shiftHint =
      this.gear === 0 || (this.gear > 1 && this.engineRpm < 1550 && vFwd < 8)
        ? -1
        : this.engineRpm > 4550 && this.gear >= 1 && this.gear < 5
          ? 1
          : 0;

    // Populate 16 sensors
    const s = this.sensorBuffer;
    s[0] = this.castTrackRay(-0.87, lateral, headingErr, curvNear);
    s[1] = this.castTrackRay(-0.43, lateral, headingErr, curvNear);
    s[2] = this.castTrackRay(0, lateral, headingErr, curvNear);
    s[3] = this.castTrackRay(0.43, lateral, headingErr, curvNear);
    s[4] = this.castTrackRay(0.87, lateral, headingErr, curvNear);
    s[5] = THREE.MathUtils.clamp(lateral / this.circuit.halfWidth, -1, 1);
    s[6] = Math.sin(headingErr);
    s[7] = Math.cos(headingErr);
    s[8] = THREE.MathUtils.clamp(curvNear * 6, -1, 1);
    s[9] = THREE.MathUtils.clamp(curvFar * 6, -1, 1);
    s[10] = THREE.MathUtils.clamp(vFwd / 32, -0.5, 1);
    s[11] = THREE.MathUtils.clamp(vLat / 8, -1, 1);
    s[12] = THREE.MathUtils.clamp((this.engineRpm - 900) / 6300, 0, 1);
    s[13] = this.gear / 5;
    s[14] = this.clutchPedal;
    s[15] = shiftHint;

    const out = this.brain.forward(s);
    const steerCmd = out[0];
    const throttle = THREE.MathUtils.clamp(out[1] * 0.5 + 0.5, 0, 1);
    const brake = THREE.MathUtils.clamp(out[2] * 0.5 + 0.5, 0, 1);
    const clutch = THREE.MathUtils.clamp(out[3] * 0.5 + 0.5, 0, 1); // 1 = pressed, 0 = released
    const shiftSignal = out[4];

    this.throttleInput = throttle;
    this.brakeInput = brake;
    this.clutchPedal = clutch;

    // Manual gear shifting logic
    if (this.shiftCooldown <= 0) {
      if (shiftSignal > 0.36 && this.gear < 5) {
        const cleanClutch = clutch > 0.28 || this.gear === 0;
        this.gear += 1;
        this.shiftCooldown = 0.28;
        if (cleanClutch) {
          this.shiftBonus += 14;
        } else {
          // Grinding gears without clutch
          this.shiftBonus -= 6;
          this.engineRpm = Math.max(700, this.engineRpm * 0.88);
        }
      } else if (shiftSignal < -0.36 && this.gear > 0) {
        const cleanClutch = clutch > 0.25;
        this.gear -= 1;
        this.shiftCooldown = 0.28;
        if (cleanClutch) {
          this.shiftBonus += 8;
        } else {
          this.shiftBonus -= 5;
        }
      }
    }

    // Manual Drivetrain: Engine RPM & Torque Curve
    const gearRatio = GEAR_RATIOS[this.gear] ?? 0;
    const totalRatio = Math.abs(gearRatio) * 3.7;
    const wheelRpm = (Math.max(0, vFwd) / (2 * Math.PI * 0.24)) * 60;
    const coupledRpm = this.gear === 0 ? 900 : wheelRpm * totalRatio;
    const engagement = this.gear === 0 ? 0 : Math.pow(1 - clutch, 1.35);

    const idleTorqueAssist = this.engineRpm < 1050 ? (1050 - this.engineRpm) * 0.0005 : 0;
    const freeRevTarget = 950 + (throttle + idleTorqueAssist) * 6100;

    // Blend RPM toward coupled wheel speed vs free revving depending on clutch engagement
    const rpmTarget = freeRevTarget * (1 - engagement * 0.85) + coupledRpm * (engagement * 0.85);
    const rpmRate = engagement > 0.6 ? 9.5 : 5.5;
    this.engineRpm = THREE.MathUtils.lerp(this.engineRpm, rpmTarget, Math.min(1, rpmRate * dt));

    // Clutch dump stall risk at very low wheel speed without throttle
    if (this.gear >= 1 && engagement > 0.72 && vFwd < 1.8 && throttle < 0.2) {
      this.engineRpm -= (480 + this.gear * 220) * dt;
    }

    if (this.age > 1.2 && this.engineRpm < 470 && this.gear >= 1 && engagement > 0.65) {
      this.stallTimer += dt;
      if (this.stallTimer > 0.22) {
        this.stalled = true;
        this.disqualify('Двигатель заглох (брошено сцепление)');
        return;
      }
    } else {
      this.stallTimer = Math.max(0, this.stallTimer - dt * 2);
    }

    this.engineRpm = THREE.MathUtils.clamp(this.engineRpm, 420, 7450);

    // Torque curve: strongest 2400..5800 RPM, rev limiter above 6900 RPM
    const normRpm = THREE.MathUtils.clamp((this.engineRpm - 800) / 5200, 0.15, 1.25);
    const revLimiter = this.engineRpm > 6850 ? 0.2 : 1.0;
    const engineTorque = (0.55 + 0.65 * Math.sin(Math.min(Math.PI, normRpm * Math.PI * 0.78))) * revLimiter;
    const driveForce =
      this.gear === 0
        ? 0
        : (throttle * 1.0 + 0.06) * engineTorque * totalRatio * engagement * 42;

    const brakeForce = brake * 480 * (vFwd > 0.2 ? 1 : 0);
    const aeroDrag = 0.28 * vFwd * Math.abs(vFwd) + 4.5 * vFwd;
    const netFwdForce = driveForce - brakeForce - aeroDrag;

    // High lateral tire grip absorbs side slip cleanly
    const latGripForce = -vLat * 210;

    this.body.applyForce(
      new CANNON.Vec3(
        fwdX * netFwdForce + rightX * latGripForce,
        0,
        fwdZ * netFwdForce + rightZ * latGripForce,
      ),
      this.body.position,
    );

    // Bicycle steering kinematics
    this.currentSteer = steerCmd * 0.54;
    const speedMag = Math.hypot(vFwd, vLat);
    if (speedMag > 0.2) {
      const yawRate = (vFwd / 1.45) * Math.tan(this.currentSteer) * 0.72;
      this.heading += yawRate * dt;
      this.body.quaternion.setFromEuler(0, this.heading, 0);
    }

    this.wheelSpin += (vFwd / 0.24) * dt;
    if (this.brakeLightMat) {
      this.brakeLightMat.emissiveIntensity = brake > 0.25 ? 2.4 : 0.45;
    }

    // Check progress stall (car refuses to launch or creeps backwards)
    if (vFwd < 0.7) {
      this.lowProgressTimer += dt;
      if (this.age > 4.0 && this.lowProgressTimer > 3.2) {
        this.disqualify('Нет разгона / пробуксовка');
        return;
      }
    } else {
      this.lowProgressTimer = 0;
    }

    // Fitness: reward circuit progress + clean manual shifting + speed in sweet RPM band - redline/curb drift
    const revPenalty = this.engineRpm > 6700 ? dt * 6 : 0;
    const centerBonus = Math.max(0, 1 - Math.abs(lateral) / this.circuit.halfWidth) * 2.5 * dt;
    const sweetRpmBonus =
      this.gear >= 1 && engagement > 0.5 && this.engineRpm >= 2100 && this.engineRpm <= 6000 && vFwd > 2
        ? dt * 4.5
        : 0;
    this.fitness =
      this.maxDistance * 16 +
      this.shiftBonus +
      Math.max(0, vFwd) * 0.6 +
      centerBonus +
      sweetRpmBonus -
      revPenalty;
  }

  private disqualify(reason: string): void {
    if (!this.alive) return;
    this.alive = false;
    this.disqualifiedReason = reason;
    this.ghosted = true;
    this.body.velocity.set(0, 0, 0);
    this.body.type = CANNON.Body.STATIC;
    this.body.mass = 0;
    this.body.updateMassProperties();
    this.body.collisionFilterGroup = 0;
    this.body.collisionFilterMask = 0;
    for (const m of this.mats) {
      m.transparent = true;
      m.opacity = 0.24;
      m.depthWrite = false;
    }
  }

  public syncVisuals(): void {
    this.meshGroup.position.set(this.body.position.x, this.body.position.y, this.body.position.z);
    this.meshGroup.quaternion.set(
      this.body.quaternion.x,
      this.body.quaternion.y,
      this.body.quaternion.z,
      this.body.quaternion.w,
    );
    for (let i = 0; i < this.frontWheelPivots.length; i++) {
      this.frontWheelPivots[i].rotation.y = this.currentSteer;
    }
    for (let i = 0; i < this.wheelMeshes.length; i++) {
      this.wheelMeshes[i].rotation.x = this.wheelSpin;
    }
  }

  public getSnapshot(): AgentSnapshot {
    const speed = Math.hypot(this.body.velocity.x, this.body.velocity.z);
    return {
      id: this.id,
      alive: this.alive,
      distance: this.distance,
      maxDistance: this.maxDistance,
      fitness: this.fitness,
      speed,
      torsoHeight: this.body.position.y,
      pitchDeg: 0,
      rollDeg: THREE.MathUtils.radToDeg(this.currentSteer),
      leftContact: true,
      rightContact: true,
      disqualifiedReason: this.alive ? undefined : this.disqualifiedReason || 'Сход с трассы',
      rpm: Math.round(this.engineRpm),
      gear: this.gear,
      clutch: Number(this.clutchPedal.toFixed(2)),
      throttle: Number(this.throttleInput.toFixed(2)),
      brake: Number(this.brakeInput.toFixed(2)),
      stalled: this.stalled,
    };
  }

  public dispose(scene: THREE.Scene): void {
    this.world.removeBody(this.body);
    scene.remove(this.rootGroup);
    this.rootGroup.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).geometry.dispose();
    });
    for (const m of this.mats) m.dispose();
  }
}

/**
 * Calibrated pretrained Manual Transmission Driver (16 -> 18 -> 5):
 * Launches in 1st with smooth clutch release, shifts cleanly 1->2->3->4->5 via tachometer,
 * steers smoothly along circuit curvature, and brakes into sharp bends.
 */
export function createPretrainedManualCarNetwork(activationFn: ActivationFn = 'tanh'): NeuralNetwork {
  const nn = new NeuralNetwork(16, 18, 5, activationFn);
  nn.wIH.fill(0);
  nn.bH.fill(0);
  nn.wHO.fill(0);
  nn.bO.fill(0);

  const setIH = (h: number, i: number, val: number) => {
    nn.wIH[h * 16 + i] = val;
  };
  const setHO = (o: number, h: number, val: number) => {
    nn.wHO[o * 18 + h] = val;
  };

  // Out 0: Steering (center track + align heading + lookahead curvature)
  // s[5] = +lateral (car right of center -> steer left +heading), s[6] = sin(headingErr), s[8] = curvNear, s[9] = curvFar
  setIH(0, 5, 1.45);  // steer left (+heading) if right of centerline (+lateral)
  setIH(0, 6, -1.65); // counter heading error
  setIH(0, 8, 1.25);  // follow near curve
  setIH(0, 9, 0.85);  // anticipate far curve
  setIH(0, 0, 0.25);
  setIH(0, 4, -0.25);
  setHO(0, 0, 1.65);

  // Out 1: Throttle (high on straights, feather when curving fast or during shift)
  setIH(1, 8, -0.45);
  setIH(1, 9, -0.35);
  setIH(1, 10, -0.35); // slight speed cap around corners
  nn.bH[1] = 1.4;
  setHO(1, 1, 1.4);

  // Out 2: Brake (only when speed high AND sharp upcoming turn)
  setIH(2, 10, 1.1);
  nn.bH[2] = -0.95;
  setHO(2, 2, 1.6);
  // Couple absolute curvature via positive/negative hidden detectors
  setIH(3, 8, 2.2);
  setIH(3, 10, 0.9);
  nn.bH[3] = -1.1;
  setHO(2, 3, 1.2);
  setIH(4, 8, -2.2);
  setIH(4, 10, 0.9);
  nn.bH[4] = -1.1;
  setHO(2, 4, 1.2);
  nn.bO[2] = -0.95;

  // Out 3: Clutch pedal (1.0 = pressed, 0.0 = engaged)
  // Default engaged (~0.1), press clutch briefly when tachometer shiftHint != 0
  setIH(5, 10, -4.0); // subtle slip only at near-standstill
  nn.bH[5] = 0.15;
  setHO(3, 5, 0.4);

  // Clutch pulse when shiftHint == +1 (upshift) or -1 (downshift)
  setIH(6, 15, 2.4);
  nn.bH[6] = -1.2;
  setHO(3, 6, 1.3);
  setIH(7, 15, -2.4);
  nn.bH[7] = -1.2;
  setHO(3, 7, 1.3);
  nn.bO[3] = -1.3;

  // Out 4: Gear Shift Lever (>0.36 upshift, <-0.36 downshift)
  // H8 = tanh(4 * shiftHint): +1 when upshift needed (RPM>5300), 0 when normal, -1 when downshift/neutral
  setIH(8, 15, 4.0);
  setHO(4, 8, 0.95);

  // H9: Neutral launch detector (+1 when gear==0 (s[13]==0), -1 when gear>=1 (s[13]>=0.2))
  // Add +1 offset via bO[4] so H9 contributes +2.0 in Neutral and 0 in gears 1..5!
  setIH(9, 13, -12.0);
  nn.bH[9] = 1.2;
  setHO(4, 9, 1.0);
  nn.bO[4] = 1.0;

  return nn;
}
