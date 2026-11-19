import * as CANNON from 'cannon-es';
import * as THREE from 'three';
import { NeuralNetwork } from '../ai/NeuralNetwork';
import { COLLISION_GROUP_ROBOT, COLLISION_GROUP_WORLD } from './BipedWalker3D';
import type { ProceduralAssets } from './materials';
import type { SimpleModeAgent } from './OtherModes';
import type { ActivationFn, AgentSnapshot } from './types';

export const DRONE_INPUT_LABELS = [
  'До кольца ΔX',
  'До кольца ΔY',
  'До кольца ΔZ',
  'Скорость VX',
  'Скорость VY',
  'Скорость VZ',
  'Тангаж X',
  'Крен Z',
  'Рыскание Y',
  'Угл. ск. X',
  'Угл. ск. Y',
  'Угл. ск. Z',
];

export const DRONE_OUTPUT_LABELS = [
  'Винт ПЛ (FL)',
  'Винт ПП (FR)',
  'Винт ЗЛ (RL)',
  'Винт ЗП (RR)',
];

const _droneQuat = new THREE.Quaternion();
const _droneEuler = new THREE.Euler();

export class DroneAgent3D implements SimpleModeAgent {
  public readonly id: number;
  public readonly brain: NeuralNetwork;
  public alive = true;
  public fitness = 0;
  public distance = 0;
  public maxDistance = 0;
  public readonly sensorBuffer = new Float32Array(12);
  public readonly rootGroup = new THREE.Group();

  private readonly body: CANNON.Body;
  private readonly world: CANNON.World;
  private readonly checkpoints: THREE.Vector3[];
  private checkpointIdx = 0;
  private checkpointsPassed = 0;
  private age = 0;
  private propSpin = 0;
  private ghosted = false;
  private disqualifiedReason = '';
  private readonly mats: THREE.MeshStandardMaterial[] = [];
  private coreMat!: THREE.MeshStandardMaterial;
  private meshGroup: THREE.Group;
  private props: THREE.Mesh[] = [];

  constructor(
    id: number,
    brain: NeuralNetwork,
    world: CANNON.World,
    physicsMat: CANNON.Material,
    assets: ProceduralAssets | null,
    checkpoints: THREE.Vector3[],
    headless = false,
  ) {
    this.id = id;
    this.brain = brain;
    this.world = world;
    this.checkpoints = checkpoints;

    this.body = new CANNON.Body({
      mass: 1.6,
      position: new CANNON.Vec3(0, 2.6, -2.5),
      material: physicsMat,
      linearDamping: 0.14,
      angularDamping: 0.35,
      collisionFilterGroup: COLLISION_GROUP_ROBOT,
      collisionFilterMask: COLLISION_GROUP_WORLD,
    });
    this.body.addShape(new CANNON.Box(new CANNON.Vec3(0.32, 0.09, 0.32)));
    this.body.allowSleep = false;

    this.body.addEventListener('collide', () => {
      if (!this.alive) return;
      this.disqualify('Столкновение с землёй');
    });

    world.addBody(this.body);

    this.meshGroup = new THREE.Group();
    if (!headless && assets) {
      this.buildDroneVisuals(assets);
      this.syncVisuals();
    }
  }

  private buildDroneVisuals(assets: ProceduralAssets): void {
    const frameMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      map: assets.carbonTexture,
      roughness: 0.55,
      metalness: 0.35,
    });
    const alloyMat = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      map: assets.brushedMetalTexture,
      roughness: 0.25,
      metalness: 0.85,
    });
    const propMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      transparent: true,
      opacity: 0.72,
      roughness: 0.3,
    });
    this.coreMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 1.6,
      roughness: 0.2,
    });
    this.mats.push(frameMat, alloyMat, propMat, this.coreMat);

    // Central avionics hub
    const hub = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.11, 0.3), alloyMat);
    hub.castShadow = true;
    this.meshGroup.add(hub);

    const visor = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.05, 0.06), this.coreMat);
    visor.position.set(0, 0.02, 0.16);
    this.meshGroup.add(visor);

    // 2 diagonal X-arms + 4 motor pods + spinning rotors
    for (const rotY of [Math.PI / 4, -Math.PI / 4]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.035, 0.78), frameMat);
      arm.rotation.y = rotY;
      arm.castShadow = true;
      this.meshGroup.add(arm);
    }

    const podCoords: Array<[number, number]> = [
      [0.27, 0.27],
      [-0.27, 0.27],
      [0.27, -0.27],
      [-0.27, -0.27],
    ];
    for (const [px, pz] of podCoords) {
      const pod = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.05, 0.08, 12), alloyMat);
      pod.position.set(px, 0.02, pz);
      this.meshGroup.add(pod);

      const led = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.012, 8, 16), this.coreMat);
      led.rotation.x = Math.PI / 2;
      led.position.set(px, 0.06, pz);
      this.meshGroup.add(led);

      const blade = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.008, 0.036), propMat);
      blade.position.set(px, 0.075, pz);
      this.meshGroup.add(blade);
      this.props.push(blade);
    }

    this.rootGroup.add(this.meshGroup);
  }

  public getTrackPosition(): THREE.Vector3 {
    return new THREE.Vector3(this.body.position.x, this.body.position.y, this.body.position.z);
  }

  public setHighlighted(v: boolean): void {
    if (this.ghosted || !this.coreMat) return;
    this.coreMat.color.setHex(v ? 0xf59e0b : 0x38bdf8);
    this.coreMat.emissive.setHex(v ? 0xd97706 : 0x0284c7);
    this.coreMat.emissiveIntensity = v ? 2.2 : 1.1;
  }

  public stepControl(dt: number): void {
    if (!this.alive) return;
    this.age += dt;

    _droneQuat.set(
      this.body.quaternion.x,
      this.body.quaternion.y,
      this.body.quaternion.z,
      this.body.quaternion.w,
    );
    _droneEuler.setFromQuaternion(_droneQuat, 'XYZ');
    const pitch = _droneEuler.x;
    const yaw = _droneEuler.y;
    const roll = _droneEuler.z;

    if (this.body.position.y < 0.25 || this.body.position.y > 18) {
      this.disqualify('Выход по высоте');
      return;
    }
    if (Math.abs(pitch) > 1.35 || Math.abs(roll) > 1.35) {
      this.disqualify('Переворот коптера');
      return;
    }
    if (Math.abs(this.body.position.x) > 18) {
      this.disqualify('Выход за коридор');
      return;
    }

    const target = this.checkpoints[this.checkpointIdx % this.checkpoints.length];
    const dx = target.x - this.body.position.x;
    const dy = target.y - this.body.position.y;
    const dz = target.z - this.body.position.z;
    const distToRing = Math.hypot(dx, dy, dz);

    if (distToRing < 1.35) {
      this.checkpointsPassed += 1;
      this.checkpointIdx += 1;
    }

    const s = this.sensorBuffer;
    s[0] = THREE.MathUtils.clamp(dx / 10, -1, 1);
    s[1] = THREE.MathUtils.clamp(dy / 6, -1, 1);
    s[2] = THREE.MathUtils.clamp(dz / 10, -1, 1);
    s[3] = THREE.MathUtils.clamp(this.body.velocity.x / 8, -1, 1);
    s[4] = THREE.MathUtils.clamp(this.body.velocity.y / 6, -1, 1);
    s[5] = THREE.MathUtils.clamp(this.body.velocity.z / 8, -1, 1);
    s[6] = THREE.MathUtils.clamp(pitch / (Math.PI * 0.5), -1, 1);
    s[7] = THREE.MathUtils.clamp(roll / (Math.PI * 0.5), -1, 1);
    s[8] = THREE.MathUtils.clamp(yaw / Math.PI, -1, 1);
    s[9] = THREE.MathUtils.clamp(this.body.angularVelocity.x / 6, -1, 1);
    s[10] = THREE.MathUtils.clamp(this.body.angularVelocity.y / 6, -1, 1);
    s[11] = THREE.MathUtils.clamp(this.body.angularVelocity.z / 6, -1, 1);

    const out = this.brain.forward(s);
    const hoverPerMotor = (this.body.mass * 9.81) / 4;

    // 4 rotor thrust multipliers mapped around hover thrust
    const fl = THREE.MathUtils.clamp(1.0 + out[0] * 0.95, 0.1, 2.1) * hoverPerMotor;
    const fr = THREE.MathUtils.clamp(1.0 + out[1] * 0.95, 0.1, 2.1) * hoverPerMotor;
    const rl = THREE.MathUtils.clamp(1.0 + out[2] * 0.95, 0.1, 2.1) * hoverPerMotor;
    const rr = THREE.MathUtils.clamp(1.0 + out[3] * 0.95, 0.1, 2.1) * hoverPerMotor;

    const motorCoords: Array<[number, number, number]> = [
      [0.27, 0.27, fl],
      [-0.27, 0.27, fr],
      [0.27, -0.27, rl],
      [-0.27, -0.27, rr],
    ];
    for (const [lx, lz, thrust] of motorCoords) {
      const localForce = new CANNON.Vec3(0, thrust, 0);
      const localPos = new CANNON.Vec3(lx, 0, lz);
      this.body.applyLocalForce(localForce, localPos);
    }

    // Counter-torque yaw stabilization (keeps heading forward +Z smoothly)
    this.body.torque.y += THREE.MathUtils.clamp(-yaw * 2.4 - this.body.angularVelocity.y * 0.9, -2.5, 2.5);

    this.propSpin += (fl + fr + rl + rr) * 2.2 * dt;

    this.distance = Math.max(0, this.body.position.z + 2.5);
    if (this.distance > this.maxDistance) this.maxDistance = this.distance;

    const ringBonus = this.checkpointsPassed * 180;
    const proximityBonus = Math.max(0, 14 - distToRing) * 4;
    const stabilityBonus = (1 - Math.min(1, (Math.abs(pitch) + Math.abs(roll)) * 0.7)) * 2.5;
    this.fitness = ringBonus + this.maxDistance * 8 + proximityBonus + this.age * stabilityBonus;
  }

  private disqualify(reason: string): void {
    if (!this.alive) return;
    this.alive = false;
    this.disqualifiedReason = reason;
    this.ghosted = true;
    this.body.velocity.set(0, 0, 0);
    this.body.angularVelocity.set(0, 0, 0);
    this.body.type = CANNON.Body.STATIC;
    this.body.mass = 0;
    this.body.updateMassProperties();
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
    this.meshGroup.quaternion.set(
      this.body.quaternion.x,
      this.body.quaternion.y,
      this.body.quaternion.z,
      this.body.quaternion.w,
    );
    for (let i = 0; i < this.props.length; i++) {
      this.props[i].rotation.y = this.propSpin * (i % 2 === 0 ? 1 : -1);
    }
  }

  public getSnapshot(): AgentSnapshot {
    _droneQuat.set(
      this.body.quaternion.x,
      this.body.quaternion.y,
      this.body.quaternion.z,
      this.body.quaternion.w,
    );
    _droneEuler.setFromQuaternion(_droneQuat, 'XYZ');
    return {
      id: this.id,
      alive: this.alive,
      distance: this.distance,
      maxDistance: this.maxDistance,
      fitness: this.fitness,
      speed: Math.hypot(this.body.velocity.x, this.body.velocity.y, this.body.velocity.z),
      torsoHeight: this.body.position.y,
      pitchDeg: THREE.MathUtils.radToDeg(_droneEuler.x),
      rollDeg: THREE.MathUtils.radToDeg(_droneEuler.z),
      leftContact: true,
      rightContact: true,
      disqualifiedReason: this.alive ? undefined : this.disqualifiedReason || 'Авария',
      checkpointsPassed: this.checkpointsPassed,
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
 * Calibrated quadcopter attitude & 3D waypoint controller (12 -> 12 -> 4).
 */
export function createPretrainedDroneNetwork(activationFn: ActivationFn = 'tanh'): NeuralNetwork {
  const nn = new NeuralNetwork(12, 12, 4, activationFn);
  nn.wIH.fill(0);
  nn.bH.fill(0);
  nn.wHO.fill(0);
  nn.bO.fill(0);

  const setIH = (h: number, i: number, val: number) => {
    nn.wIH[h * 12 + i] = val;
  };
  const setHO = (o: number, h: number, val: number) => {
    nn.wHO[o * 12 + h] = val;
  };

  // Rotor mapping reminder:
  // FL = (+0.27, +0.27), FR = (-0.27, +0.27), RL = (+0.27, -0.27), RR = (-0.27, -0.27)
  // More thrust on front (+Z: FL, FR) pitches nose UP (-pitch in +X euler -> accelerates -Z).
  // More thrust on rear (-Z: RL, RR) pitches nose DOWN (+pitch -> accelerates +Z).
  // More thrust on left (+X: FL, RL) rolls right (-roll around Z -> accelerates -X).
  // Wait: torque = r x F = (lx, 0, lz) x (0, F, 0) = (-lz*F, 0, +lx*F)!
  // Let's double check cross product: (0,0,+0.27) x (0,F,0) = (-0.27*F, 0, 0) -> torque X negative!
  // (+0.27, 0, 0) x (0, F, 0) = (0, 0, +0.27*F) -> torque Z positive!
  // And rotating body around +X gives nose moving DOWN (-Y, +Z? Wait: right hand rule around +X: +Y rotates to +Z!).
  // So +roll around +Z moves +Y toward -X! Wait: (0,1,0) rotated +theta around +Z = (-sin theta, cos theta, 0), so +roll accelerates -X!

  // H0: Collective altitude hold (ΔY = s[1], VY = s[4])
  setIH(0, 1, 1.3);
  setIH(0, 4, -0.9);
  for (let o = 0; o < 4; o++) setHO(o, 0, 0.85);

  // H1: Desired +Z acceleration via positive pitch (+theta around X rotates +Y toward +Z)
  // Wait: target forward ΔZ = s[2], VZ = s[5] -> desired pitch = +0.22*(s[2]*1.2 - s[5])
  // Pitch error = desired_pitch - s[6]. Need +torqueX when pitch error > 0 -> increase Rear (lz<0 -> RL,RR) and decrease Front (FL,FR)
  setIH(1, 2, 0.36);  // +ΔZ -> pitch forward
  setIH(1, 5, -0.38); // damp VZ
  setIH(1, 6, -1.55); // pitch feedback
  setIH(1, 9, -0.45); // angular velocity X damping
  setHO(0, 1, -0.9);  // FL
  setHO(1, 1, -0.9);  // FR
  setHO(2, 1, 0.9);   // RL
  setHO(3, 1, 0.9);   // RR

  // H2: Desired +X acceleration -> requires -roll (since +roll rotates +Y toward -X)
  // Wait: +torqueZ comes from +lx (FL, RL). To create -roll (for +X motion), we need -torqueZ -> increase FR, RR (lx<0) and decrease FL, RL!
  setIH(2, 0, 0.36);  // +ΔX -> negative roll -> more FR, RR
  setIH(2, 3, -0.38); // damp VX
  setIH(2, 7, 1.55);  // if roll > 0, need -torqueZ -> more FR, RR? Wait: if roll > 0, to return roll->0 we need -torqueZ -> yes, more FR/RR!
  setIH(2, 11, 0.45); // angular velocity Z damping
  setHO(0, 2, -0.9);  // FL
  setHO(1, 2, 0.9);   // FR
  setHO(2, 2, -0.9);  // RL
  setHO(3, 2, 0.9);   // RR

  return nn;
}
