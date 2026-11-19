import * as CANNON from 'cannon-es';
import * as THREE from 'three';
import { NeuralNetwork } from '../ai/NeuralNetwork';
import {
  COLLISION_GROUP_ROBOT,
  COLLISION_GROUP_WORLD,
} from './BipedWalker3D';
import type { ProceduralAssets } from './materials';
import type { ActivationFn, AgentSnapshot } from './types';

export const ROVER_INPUT_LABELS = [
  'Луч -50°',
  'Луч -25°',
  'Луч 0°',
  'Луч +25°',
  'Луч +50°',
  'Цель sin(θ)',
  'Цель cos(θ)',
  'Дистанция до цели',
  'Скорость',
];
export const ROVER_OUTPUT_LABELS = ['Тяга', 'Руль'];

export const BALANCE_INPUT_LABELS = [
  'Позиция тележки',
  'Скорость тележки',
  'Угол маятника',
  'Угл. скорость маятника',
  'Импульс ветра',
  'Отклонение от центра',
];
export const BALANCE_OUTPUT_LABELS = ['Привод тележки'];

export const PARKOUR_INPUT_LABELS = [
  'До барьера',
  'Высота барьера',
  'Контакт с полом',
  'Вертик. скорость',
  'Скорость вперёд',
  'Отклонение X',
  'Наклон корпуса',
  'Ритм прыжка',
];
export const PARKOUR_OUTPUT_LABELS = ['Ускорение', 'Прыжок', 'Манёвр X'];

export interface SimpleModeAgent {
  id: number;
  brain: NeuralNetwork;
  alive: boolean;
  fitness: number;
  distance: number;
  maxDistance: number;
  sensorBuffer: Float32Array;
  rootGroup: THREE.Group;
  getTrackPosition: () => THREE.Vector3;
  setHighlighted: (v: boolean) => void;
  stepControl: (dt: number) => void;
  syncVisuals: () => void;
  getSnapshot: () => AgentSnapshot;
  dispose: (scene: THREE.Scene) => void;
}

export class RoverAgent3D implements SimpleModeAgent {
  public readonly id: number;
  public readonly brain: NeuralNetwork;
  public alive = true;
  public fitness = 0;
  public distance = 0;
  public maxDistance = 0;
  public readonly sensorBuffer = new Float32Array(9);
  public readonly rootGroup = new THREE.Group();

  private readonly body: CANNON.Body;
  private readonly world: CANNON.World;
  private readonly obstacles: Array<{ x: number; z: number; r: number }>;
  private readonly beacons: THREE.Vector3[];
  private beaconIdx = 0;
  private beaconsCollected = 0;
  private age = 0;
  private heading = 0;
  private ghosted = false;
  private readonly mats: THREE.MeshStandardMaterial[] = [];
  private coreMat!: THREE.MeshStandardMaterial;
  private meshGroup: THREE.Group;

  constructor(
    id: number,
    brain: NeuralNetwork,
    world: CANNON.World,
    physicsMat: CANNON.Material,
    assets: ProceduralAssets,
    obstacles: Array<{ x: number; z: number; r: number }>,
    beacons: THREE.Vector3[],
  ) {
    this.id = id;
    this.brain = brain;
    this.world = world;
    this.obstacles = obstacles;
    this.beacons = beacons;

    this.body = new CANNON.Body({
      mass: 6,
      position: new CANNON.Vec3(0, 0.28, -6),
      material: physicsMat,
      linearDamping: 0.35,
      angularDamping: 0.6,
      collisionFilterGroup: COLLISION_GROUP_ROBOT,
      collisionFilterMask: COLLISION_GROUP_WORLD,
    });
    this.body.addShape(new CANNON.Box(new CANNON.Vec3(0.32, 0.14, 0.45)));
    this.body.allowSleep = false;
    this.body.addEventListener('collide', (e: { body: CANNON.Body }) => {
      if (!this.alive) return;
      if (e.body.mass === 0 && e.body.position.y > 0.1) {
        this.disqualify('Столкновение с колонной');
      }
    });
    world.addBody(this.body);

    // Composite 6-part rover mesh
    this.meshGroup = new THREE.Group();
    const hullMat = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      map: assets.brushedMetalTexture,
      roughness: 0.3,
      metalness: 0.8,
    });
    const wheelMat = new THREE.MeshStandardMaterial({
      color: 0x111827,
      roughness: 0.9,
      metalness: 0.1,
    });
    this.coreMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 1.4,
      roughness: 0.2,
    });
    this.mats.push(hullMat, wheelMat, this.coreMat);

    const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.18, 0.84), hullMat);
    chassis.castShadow = true;
    this.meshGroup.add(chassis);

    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 0.22, 12), this.coreMat);
    mast.position.set(0, 0.16, 0.16);
    this.meshGroup.add(mast);

    const wheelGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.1, 14);
    wheelGeo.rotateZ(Math.PI / 2);
    for (const [wx, wz] of [
      [-0.31, 0.28],
      [0.31, 0.28],
      [-0.31, -0.28],
      [0.31, -0.28],
    ]) {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.position.set(wx, -0.06, wz);
      w.castShadow = true;
      this.meshGroup.add(w);
    }

    this.rootGroup.add(this.meshGroup);
    this.syncVisuals();
  }

  public getTrackPosition(): THREE.Vector3 {
    return new THREE.Vector3(this.body.position.x, this.body.position.y, this.body.position.z);
  }

  public setHighlighted(v: boolean): void {
    if (this.ghosted) return;
    this.coreMat.color.setHex(v ? 0xf59e0b : 0x38bdf8);
    this.coreMat.emissive.setHex(v ? 0xd97706 : 0x0284c7);
    this.coreMat.emissiveIntensity = v ? 2.2 : 1.1;
  }

  private castRayDistance(angleOffset: number): number {
    const dirX = Math.sin(this.heading + angleOffset);
    const dirZ = Math.cos(this.heading + angleOffset);
    const maxDist = 7.5;
    let minD = maxDist;

    const px = this.body.position.x;
    const pz = this.body.position.z;

    for (let i = 0; i < this.obstacles.length; i++) {
      const obs = this.obstacles[i];
      const vx = obs.x - px;
      const vz = obs.z - pz;
      const proj = vx * dirX + vz * dirZ;
      if (proj < 0 || proj > minD) continue;
      const perpSq = vx * vx + vz * vz - proj * proj;
      const r = obs.r + 0.35;
      if (perpSq < r * r) {
        const d = proj - Math.sqrt(r * r - perpSq);
        if (d < minD && d > 0) minD = d;
      }
    }
    return minD / maxDist;
  }

  public stepControl(dt: number): void {
    if (!this.alive) return;
    this.age += dt;

    if (Math.abs(this.body.position.x) > 9.5 || Math.abs(this.body.position.z) > 13.5) {
      this.disqualify('Выход за границы полигона');
      return;
    }

    const target = this.beacons[this.beaconIdx % this.beacons.length];
    const dx = target.x - this.body.position.x;
    const dz = target.z - this.body.position.z;
    const dist = Math.hypot(dx, dz);

    if (dist < 1.1) {
      this.beaconsCollected++;
      this.beaconIdx++;
    }

    const targetAngle = Math.atan2(dx, dz);
    const relAngle = targetAngle - this.heading;
    const speed = Math.hypot(this.body.velocity.x, this.body.velocity.z);

    const s = this.sensorBuffer;
    s[0] = this.castRayDistance(-0.87);
    s[1] = this.castRayDistance(-0.43);
    s[2] = this.castRayDistance(0);
    s[3] = this.castRayDistance(0.43);
    s[4] = this.castRayDistance(0.87);
    s[5] = Math.sin(relAngle);
    s[6] = Math.cos(relAngle);
    s[7] = THREE.MathUtils.clamp(dist / 16, 0, 1);
    s[8] = THREE.MathUtils.clamp(speed / 6, 0, 1);

    const out = this.brain.forward(s);
    const throttle = (out[0] * 0.5 + 0.5) * 44;
    const steer = out[1] * 3.4;

    this.heading += steer * dt;
    this.body.quaternion.setFromEuler(0, this.heading, 0);
    this.body.applyForce(
      new CANNON.Vec3(Math.sin(this.heading) * throttle, 0, Math.cos(this.heading) * throttle),
      this.body.position,
    );

    this.distance = this.beaconsCollected * 10 + Math.max(0, 12 - dist);
    if (this.distance > this.maxDistance) this.maxDistance = this.distance;
    this.fitness = this.beaconsCollected * 140 + Math.max(0, 14 - dist) * 5 + this.age * 0.4;
  }

  private disqualify(reason: string): void {
    if (!this.alive) return;
    this.alive = false;
    this.ghosted = true;
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
    void reason;
  }

  public syncVisuals(): void {
    this.meshGroup.position.set(this.body.position.x, this.body.position.y, this.body.position.z);
    this.meshGroup.quaternion.set(
      this.body.quaternion.x,
      this.body.quaternion.y,
      this.body.quaternion.z,
      this.body.quaternion.w,
    );
  }

  public getSnapshot(): AgentSnapshot {
    return {
      id: this.id,
      alive: this.alive,
      distance: this.distance,
      maxDistance: this.maxDistance,
      fitness: this.fitness,
      speed: Math.hypot(this.body.velocity.x, this.body.velocity.z),
      torsoHeight: this.body.position.y,
      pitchDeg: 0,
      rollDeg: 0,
      leftContact: true,
      rightContact: true,
      disqualifiedReason: this.alive ? undefined : 'Столкновение / граница',
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

export class CartPoleAgent3D implements SimpleModeAgent {
  public readonly id: number;
  public readonly brain: NeuralNetwork;
  public alive = true;
  public fitness = 0;
  public distance = 0;
  public maxDistance = 0;
  public readonly sensorBuffer = new Float32Array(6);
  public readonly rootGroup = new THREE.Group();

  private readonly cart: CANNON.Body;
  private readonly pole: CANNON.Body;
  private readonly hinge: CANNON.HingeConstraint;
  private readonly world: CANNON.World;
  private readonly laneX: number;
  private age = 0;
  private wheelSpin = 0;
  private movementAccum = 0;
  private ghosted = false;
  private readonly mats: THREE.MeshStandardMaterial[] = [];
  private coreMat!: THREE.MeshStandardMaterial;
  private cartMesh: THREE.Group;
  private poleMesh: THREE.Group;
  private wheels: THREE.Mesh[] = [];

  constructor(
    id: number,
    brain: NeuralNetwork,
    world: CANNON.World,
    physicsMat: CANNON.Material,
    assets: ProceduralAssets,
    laneX: number,
  ) {
    this.id = id;
    this.brain = brain;
    this.world = world;
    this.laneX = laneX;

    this.cart = new CANNON.Body({
      mass: 4.5,
      position: new CANNON.Vec3(laneX, 0.24, 0),
      material: physicsMat,
      linearDamping: 0.04,
      angularDamping: 0.99,
      collisionFilterGroup: 0,
      collisionFilterMask: 0,
    });
    this.cart.addShape(new CANNON.Box(new CANNON.Vec3(0.32, 0.14, 0.45)));
    this.cart.fixedRotation = true;
    this.cart.updateMassProperties();
    this.cart.allowSleep = false;

    // Distinct non-zero initial tilt so carts MUST actively move right from frame 1
    const sign = id % 2 === 0 ? 1 : -1;
    const initialTilt = sign * (0.09 + Math.random() * 0.11);
    this.pole = new CANNON.Body({
      mass: 1.4,
      position: new CANNON.Vec3(laneX, 1.14, Math.sin(initialTilt) * 0.8),
      material: physicsMat,
      linearDamping: 0.015,
      angularDamping: 0.02,
      collisionFilterGroup: 0,
      collisionFilterMask: 0,
    });
    this.pole.addShape(new CANNON.Box(new CANNON.Vec3(0.045, 0.8, 0.045)));
    this.pole.quaternion.setFromEuler(initialTilt, 0, 0);
    this.pole.allowSleep = false;

    this.hinge = new CANNON.HingeConstraint(this.cart, this.pole, {
      pivotA: new CANNON.Vec3(0, 0.14, 0),
      axisA: new CANNON.Vec3(1, 0, 0),
      pivotB: new CANNON.Vec3(0, -0.8, 0),
      axisB: new CANNON.Vec3(1, 0, 0),
      collideConnected: false,
    });

    world.addBody(this.cart);
    world.addBody(this.pole);
    world.addConstraint(this.hinge);

    const metalMat = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      map: assets.brushedMetalTexture,
      roughness: 0.28,
      metalness: 0.82,
    });
    const carbonMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      map: assets.carbonTexture,
      roughness: 0.6,
      metalness: 0.25,
    });
    const wheelMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.85,
      metalness: 0.2,
    });
    this.coreMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 1.4,
      roughness: 0.2,
    });
    this.mats.push(metalMat, carbonMat, wheelMat, this.coreMat);

    this.cartMesh = new THREE.Group();
    const base = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.2, 0.86), carbonMat);
    base.castShadow = true;
    this.cartMesh.add(base);
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.22, 12), this.coreMat);
    hub.rotation.z = Math.PI / 2;
    hub.position.set(0, 0.14, 0);
    this.cartMesh.add(hub);

    const wheelGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.08, 12);
    wheelGeo.rotateZ(Math.PI / 2);
    for (const [wx, wz] of [
      [-0.31, 0.3],
      [0.31, 0.3],
      [-0.31, -0.3],
      [0.31, -0.3],
    ]) {
      const wMesh = new THREE.Mesh(wheelGeo, wheelMat);
      wMesh.position.set(wx, -0.1, wz);
      wMesh.castShadow = true;
      this.cartMesh.add(wMesh);
      this.wheels.push(wMesh);
    }

    this.poleMesh = new THREE.Group();
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.045, 1.56, 12), metalMat);
    rod.castShadow = true;
    this.poleMesh.add(rod);
    const tipWeight = new THREE.Mesh(new THREE.SphereGeometry(0.11, 14, 12), this.coreMat);
    tipWeight.position.set(0, 0.78, 0);
    this.poleMesh.add(tipWeight);

    this.rootGroup.add(this.cartMesh);
    this.rootGroup.add(this.poleMesh);
    this.syncVisuals();
  }

  public getTrackPosition(): THREE.Vector3 {
    return new THREE.Vector3(this.cart.position.x, 0.8, this.cart.position.z);
  }

  public setHighlighted(v: boolean): void {
    if (this.ghosted) return;
    this.coreMat.color.setHex(v ? 0xf59e0b : 0x38bdf8);
    this.coreMat.emissive.setHex(v ? 0xd97706 : 0x0284c7);
    this.coreMat.emissiveIntensity = v ? 2.2 : 1.1;
  }

  public stepControl(dt: number): void {
    if (!this.alive) return;
    this.age += dt;

    // Lock cart strictly on frictionless Z rail at laneX, y = 0.24
    this.cart.position.x = this.laneX;
    this.cart.position.y = 0.24;
    this.cart.velocity.x = 0;
    this.cart.velocity.y = 0;

    // Apply periodic wind gusts on pole tip so cart must continuously move to stay balanced
    const gustWave = Math.sin(this.age * 2.1 + this.id * 0.6);
    const windImpulse = Math.abs(gustWave) > 0.75 ? Math.sign(gustWave) * 1.15 : 0;
    if (windImpulse !== 0) {
      this.pole.applyForce(new CANNON.Vec3(0, 0, windImpulse), this.pole.position);
    }

    const q = new THREE.Quaternion(
      this.pole.quaternion.x,
      this.pole.quaternion.y,
      this.pole.quaternion.z,
      this.pole.quaternion.w,
    );
    const euler = new THREE.Euler().setFromQuaternion(q, 'XYZ');
    const tilt = euler.x;

    if (Math.abs(tilt) > 0.72 || Math.abs(this.cart.position.z) > 6.5) {
      this.disqualify();
      return;
    }

    const s = this.sensorBuffer;
    s[0] = THREE.MathUtils.clamp(this.cart.position.z / 6, -1, 1);
    s[1] = THREE.MathUtils.clamp(this.cart.velocity.z / 7, -1, 1);
    s[2] = THREE.MathUtils.clamp(tilt / 0.72, -1, 1);
    s[3] = THREE.MathUtils.clamp(this.pole.angularVelocity.x / 6, -1, 1);
    s[4] = THREE.MathUtils.clamp(windImpulse / 1.5, -1, 1);
    s[5] = THREE.MathUtils.clamp(-this.cart.position.z / 6, -1, 1);

    const out = this.brain.forward(s);
    const force = out[0] * 145;
    this.cart.applyForce(new CANNON.Vec3(0, 0, force), this.cart.position);

    this.wheelSpin += (this.cart.velocity.z / 0.11) * dt;
    this.movementAccum += Math.abs(this.cart.velocity.z) * dt;

    // Distance metric reflects active stabilization travel + survival
    this.distance = this.age * 1.6 + this.movementAccum * 0.8;
    this.maxDistance = this.distance;

    // Reward upright pole + active corrective travel + rail centering
    const uprightScore = Math.max(0, 1.0 - Math.abs(tilt) * 1.8);
    const activeMotionBonus = Math.min(2.2, Math.abs(this.cart.velocity.z) * 0.7);
    const centerPenalty = Math.abs(this.cart.position.z) * 0.45;
    this.fitness += dt * (8.5 * uprightScore + activeMotionBonus - centerPenalty);
  }

  private disqualify(): void {
    this.alive = false;
    this.ghosted = true;
    this.world.removeConstraint(this.hinge);
    this.cart.type = CANNON.Body.STATIC;
    this.pole.type = CANNON.Body.STATIC;
    this.cart.mass = 0;
    this.pole.mass = 0;
    for (const m of this.mats) {
      m.transparent = true;
      m.opacity = 0.22;
      m.depthWrite = false;
    }
  }

  public syncVisuals(): void {
    this.cartMesh.position.set(this.cart.position.x, this.cart.position.y, this.cart.position.z);
    for (let i = 0; i < this.wheels.length; i++) {
      this.wheels[i].rotation.x = this.wheelSpin;
    }
    this.poleMesh.position.set(this.pole.position.x, this.pole.position.y, this.pole.position.z);
    this.poleMesh.quaternion.set(
      this.pole.quaternion.x,
      this.pole.quaternion.y,
      this.pole.quaternion.z,
      this.pole.quaternion.w,
    );
  }

  public getSnapshot(): AgentSnapshot {
    return {
      id: this.id,
      alive: this.alive,
      distance: this.distance,
      maxDistance: this.maxDistance,
      fitness: this.fitness,
      speed: Math.abs(this.cart.velocity.z),
      torsoHeight: this.pole.position.y,
      pitchDeg: THREE.MathUtils.radToDeg(this.sensorBuffer[2] * 0.72),
      rollDeg: 0,
      leftContact: true,
      rightContact: true,
      disqualifiedReason: this.alive ? undefined : 'Маятник потерял баланс',
    };
  }

  public dispose(scene: THREE.Scene): void {
    this.world.removeConstraint(this.hinge);
    this.world.removeBody(this.cart);
    this.world.removeBody(this.pole);
    scene.remove(this.rootGroup);
    this.rootGroup.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) (o as THREE.Mesh).geometry.dispose();
    });
    for (const m of this.mats) m.dispose();
  }
}

export function createPretrainedCartPoleNetwork(activationFn: ActivationFn = 'tanh'): NeuralNetwork {
  const nn = new NeuralNetwork(6, 8, 1, activationFn);
  nn.wIH.fill(0);
  nn.bH.fill(0);
  nn.wHO.fill(0);
  nn.bO.fill(0);

  // H0: Pole PD stabilizer (+tilt -> +force in +Z to catch falling pole)
  nn.wIH[0 * 6 + 2] = 2.6;  // tilt
  nn.wIH[0 * 6 + 3] = 1.15; // pole angular vel
  nn.wIH[0 * 6 + 0] = 0.35; // gentle cart pos centering
  nn.wIH[0 * 6 + 1] = 0.45; // cart vel damping
  nn.wIH[0 * 6 + 4] = 0.25; // anticipate wind gust
  nn.wHO[0] = 2.4;

  // H1: Secondary anti-drift centering & active counter-move
  nn.wIH[1 * 6 + 2] = 1.4;
  nn.wIH[1 * 6 + 5] = -0.35;
  nn.wHO[1] = 0.9;

  return nn;
}

export class ParkourAgent3D implements SimpleModeAgent {
  public readonly id: number;
  public readonly brain: NeuralNetwork;
  public alive = true;
  public fitness = 0;
  public distance = 0;
  public maxDistance = 0;
  public readonly sensorBuffer = new Float32Array(8);
  public readonly rootGroup = new THREE.Group();

  private readonly body: CANNON.Body;
  private readonly world: CANNON.World;
  private readonly hurdles: Array<{ z: number; h: number }>;
  private age = 0;
  private grounded = false;
  private jumpCooldown = 0;
  private ghosted = false;
  private readonly mats: THREE.MeshStandardMaterial[] = [];
  private coreMat!: THREE.MeshStandardMaterial;
  private meshGroup: THREE.Group;

  constructor(
    id: number,
    brain: NeuralNetwork,
    world: CANNON.World,
    physicsMat: CANNON.Material,
    assets: ProceduralAssets,
    hurdles: Array<{ z: number; h: number }>,
  ) {
    this.id = id;
    this.brain = brain;
    this.world = world;
    this.hurdles = hurdles;

    this.body = new CANNON.Body({
      mass: 5,
      position: new CANNON.Vec3(0, 0.45, 0),
      material: physicsMat,
      linearDamping: 0.08,
      angularDamping: 0.7,
      collisionFilterGroup: COLLISION_GROUP_ROBOT,
      collisionFilterMask: COLLISION_GROUP_WORLD,
    });
    this.body.addShape(new CANNON.Box(new CANNON.Vec3(0.28, 0.28, 0.35)));
    this.body.fixedRotation = true;
    this.body.updateMassProperties();
    this.body.allowSleep = false;

    this.body.addEventListener('collide', (e: { body: CANNON.Body; contact: CANNON.ContactEquation }) => {
      if (!this.alive) return;
      if (e.body.position.y > 0.05) {
        this.disqualify();
      } else {
        this.grounded = true;
      }
    });
    world.addBody(this.body);

    const shellMat = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      map: assets.brushedMetalTexture,
      roughness: 0.3,
      metalness: 0.8,
    });
    const skidMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      map: assets.carbonTexture,
      roughness: 0.7,
      metalness: 0.2,
    });
    this.coreMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 1.5,
      roughness: 0.2,
    });
    this.mats.push(shellMat, skidMat, this.coreMat);

    this.meshGroup = new THREE.Group();
    const hull = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.38, 0.66), shellMat);
    hull.castShadow = true;
    this.meshGroup.add(hull);
    const thruster = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.18, 0.2, 12), this.coreMat);
    thruster.position.set(0, -0.14, -0.25);
    this.meshGroup.add(thruster);
    const skid = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.1, 0.7), skidMat);
    skid.position.set(0, -0.22, 0);
    this.meshGroup.add(skid);

    this.rootGroup.add(this.meshGroup);
    this.syncVisuals();
  }

  public getTrackPosition(): THREE.Vector3 {
    return new THREE.Vector3(this.body.position.x, this.body.position.y, this.body.position.z);
  }

  public setHighlighted(v: boolean): void {
    if (this.ghosted) return;
    this.coreMat.color.setHex(v ? 0xf59e0b : 0x38bdf8);
    this.coreMat.emissive.setHex(v ? 0xd97706 : 0x0284c7);
    this.coreMat.emissiveIntensity = v ? 2.2 : 1.1;
  }

  public stepControl(dt: number): void {
    if (!this.alive) return;
    this.age += dt;
    this.jumpCooldown = Math.max(0, this.jumpCooldown - dt);
    if (this.body.position.y <= 0.34) this.grounded = true;
    else if (this.body.position.y > 0.45) this.grounded = false;

    if (Math.abs(this.body.position.x) > 3.5) {
      this.disqualify();
      return;
    }

    let nextDist = 12;
    let nextH = 0;
    for (const h of this.hurdles) {
      const dz = h.z - this.body.position.z;
      if (dz > -0.2 && dz < nextDist) {
        nextDist = dz;
        nextH = h.h;
      }
    }

    const s = this.sensorBuffer;
    s[0] = THREE.MathUtils.clamp(nextDist / 12, 0, 1);
    s[1] = THREE.MathUtils.clamp(nextH / 1.0, 0, 1);
    s[2] = this.grounded ? 1 : 0;
    s[3] = THREE.MathUtils.clamp(this.body.velocity.y / 8, -1, 1);
    s[4] = THREE.MathUtils.clamp(this.body.velocity.z / 10, -1, 1);
    s[5] = THREE.MathUtils.clamp(this.body.position.x / 3.5, -1, 1);
    s[6] = THREE.MathUtils.clamp((this.body.position.y - 0.3) / 1.5, 0, 1);
    s[7] = Math.sin(this.age * 4);

    const out = this.brain.forward(s);
    const fwdForce = (out[0] * 0.5 + 0.5) * 44;
    this.body.applyForce(new CANNON.Vec3(out[2] * 12, 0, fwdForce), this.body.position);

    if (out[1] > 0.3 && this.grounded && this.jumpCooldown <= 0) {
      this.body.velocity.y = 7.6;
      this.grounded = false;
      this.jumpCooldown = 0.42;
    }

    this.distance = this.body.position.z;
    if (this.distance > this.maxDistance) this.maxDistance = this.distance;
    this.fitness = this.maxDistance * 12 + this.age * 0.5;
  }

  private disqualify(): void {
    if (!this.alive) return;
    this.alive = false;
    this.ghosted = true;
    this.body.type = CANNON.Body.STATIC;
    this.body.mass = 0;
    this.body.collisionFilterGroup = 0;
    this.body.collisionFilterMask = 0;
    for (const m of this.mats) {
      m.transparent = true;
      m.opacity = 0.22;
      m.depthWrite = false;
    }
  }

  public syncVisuals(): void {
    this.meshGroup.position.set(this.body.position.x, this.body.position.y, this.body.position.z);
  }

  public getSnapshot(): AgentSnapshot {
    return {
      id: this.id,
      alive: this.alive,
      distance: this.distance,
      maxDistance: this.maxDistance,
      fitness: this.fitness,
      speed: this.body.velocity.z,
      torsoHeight: this.body.position.y,
      pitchDeg: 0,
      rollDeg: 0,
      leftContact: this.grounded,
      rightContact: this.grounded,
      disqualifiedReason: this.alive ? undefined : 'Столкновение с барьером',
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
