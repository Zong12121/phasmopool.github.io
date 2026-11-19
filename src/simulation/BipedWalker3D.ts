import * as CANNON from 'cannon-es';
import * as THREE from 'three';
import { NeuralNetwork } from '../ai/NeuralNetwork';
import type { ProceduralAssets } from './materials';
import type { AgentSnapshot } from './types';

export const COLLISION_GROUP_WORLD = 1;
export const COLLISION_GROUP_ROBOT = 2;
export const COLLISION_GROUP_PROJECTILE = 4;

export const BIPED_INPUT_LABELS = [
  'Наклон X (тангаж)',
  'Поворот Y (рыскание)',
  'Крен Z',
  'Угл. скорость X',
  'Угл. скорость Y',
  'Угл. скорость Z',
  'Лин. скорость X',
  'Лин. скорость Y',
  'Лин. скорость Z',
  'Угол лев. бедра',
  'Угол прав. бедра',
  'Угол лев. колена',
  'Угол прав. колена',
  'Контакт лев. стопы',
  'Контакт прав. стопы',
  'Ритм спин. мозга sin',
  'Ритм спин. мозга cos',
  'Высота торса',
];

export const BIPED_OUTPUT_LABELS = [
  'Мотор лев. бедра',
  'Мотор прав. бедра',
  'Мотор лев. колена',
  'Мотор прав. колена',
  'Мотор лев. плеча',
  'Мотор прав. плеча',
];

// Reusable math objects to ensure zero GC in animation/physics loop
const _qInv = new CANNON.Quaternion();
const _qRel = new CANNON.Quaternion();
const _vLocalZ = new CANNON.Vec3();
const _vWorldZ = new CANNON.Vec3(0, 0, 1);
const _euler = new THREE.Euler();
const _threeQuat = new THREE.Quaternion();

function computeHingeAngleX(bodyA: CANNON.Body, bodyB: CANNON.Body): number {
  bodyA.quaternion.inverse(_qInv);
  _qInv.mult(bodyB.quaternion, _qRel);
  _qRel.vmult(_vWorldZ, _vLocalZ);
  return Math.atan2(-_vLocalZ.y, _vLocalZ.z);
}

export interface BipedPhysicsMaterials {
  chassisMaterial: CANNON.Material;
  footMaterial: CANNON.Material;
}

export class BipedWalker3D {
  public readonly id: number;
  public readonly brain: NeuralNetwork;
  public readonly world: CANNON.World;
  public readonly rootGroup: THREE.Group;

  // 9 Rigid Bodies
  public readonly torso: CANNON.Body;
  public readonly leftThigh: CANNON.Body;
  public readonly rightThigh: CANNON.Body;
  public readonly leftShin: CANNON.Body;
  public readonly rightShin: CANNON.Body;
  public readonly leftFoot: CANNON.Body;
  public readonly rightFoot: CANNON.Body;
  public readonly leftArm: CANNON.Body;
  public readonly rightArm: CANNON.Body;
  public readonly allBodies: CANNON.Body[];

  // Joints
  public readonly leftHip: CANNON.HingeConstraint;
  public readonly rightHip: CANNON.HingeConstraint;
  public readonly leftKnee: CANNON.HingeConstraint;
  public readonly rightKnee: CANNON.HingeConstraint;
  public readonly leftAnkle: CANNON.HingeConstraint;
  public readonly rightAnkle: CANNON.HingeConstraint;
  public readonly leftShoulder: CANNON.HingeConstraint;
  public readonly rightShoulder: CANNON.HingeConstraint;
  public readonly allConstraints: CANNON.HingeConstraint[];

  // Meshes paired with bodies
  private readonly bodyMeshes: Array<{ body: CANNON.Body; group: THREE.Group }> = [];
  private readonly clonedMaterials: THREE.MeshStandardMaterial[] = [];
  private accentMaterial!: THREE.MeshStandardMaterial;
  private coreGlowMaterial!: THREE.MeshStandardMaterial;
  private contactShadowMesh: THREE.Mesh | null = null;

  // Sensor & state tracking
  public alive = true;
  public disqualifiedReason = '';
  public age = 0;
  public distance = 0;
  public maxDistance = 0;
  public fitness = 0;
  public uprightAccumulator = 0;
  public stabilityPenalty = 0;
  public leftFootContact = false;
  public rightFootContact = false;
  private hazardBodyContact = false;
  private hazardReason = '';

  public readonly sensorBuffer = new Float32Array(18);
  private isHighlighted = false;
  private ghosted = false;
  private readonly cpgFreq = 1.85; // Hz natural brisk bipedal stepping cadence

  private readonly groundBody: CANNON.Body;

  constructor(
    id: number,
    brain: NeuralNetwork,
    world: CANNON.World,
    groundBody: CANNON.Body,
    physicsMats: BipedPhysicsMaterials,
    assets: ProceduralAssets | null,
    spawnOffset = new CANNON.Vec3(0, 0, 0),
    headless = false,
  ) {
    this.id = id;
    this.brain = brain;
    this.world = world;
    this.groundBody = groundBody;
    this.rootGroup = new THREE.Group();

    const ox = spawnOffset.x;
    const oy = spawnOffset.y;
    const oz = spawnOffset.z;

    const makeBody = (
      mass: number,
      x: number,
      y: number,
      z: number,
      material: CANNON.Material,
      angularDamping = 0.08,
    ) => {
      const b = new CANNON.Body({
        mass,
        position: new CANNON.Vec3(ox + x, oy + y, oz + z),
        material,
        linearDamping: 0.03,
        angularDamping,
        collisionFilterGroup: COLLISION_GROUP_ROBOT,
        collisionFilterMask: COLLISION_GROUP_WORLD | COLLISION_GROUP_PROJECTILE,
      });
      b.allowSleep = false;
      return b;
    };

    // 1. Torso + Pelvis + Head Sensor Pod (Compound Body)
    this.torso = makeBody(14.0, 0, 1.24, 0, physicsMats.chassisMaterial, 0.1);
    this.torso.addShape(new CANNON.Box(new CANNON.Vec3(0.17, 0.09, 0.11)), new CANNON.Vec3(0, -0.24, 0)); // Pelvis
    this.torso.addShape(new CANNON.Box(new CANNON.Vec3(0.20, 0.19, 0.13)), new CANNON.Vec3(0, 0.04, 0));  // Chest
    this.torso.addShape(new CANNON.Sphere(0.13), new CANNON.Vec3(0, 0.36, 0.03));                         // Head dome

    // 2 & 3. Thighs
    this.leftThigh = makeBody(2.6, 0.15, 0.73, 0, physicsMats.chassisMaterial, 0.06);
    this.leftThigh.addShape(new CANNON.Box(new CANNON.Vec3(0.06, 0.22, 0.065)));

    this.rightThigh = makeBody(2.6, -0.15, 0.73, 0, physicsMats.chassisMaterial, 0.06);
    this.rightThigh.addShape(new CANNON.Box(new CANNON.Vec3(0.06, 0.22, 0.065)));

    // 4 & 5. Shins
    this.leftShin = makeBody(1.6, 0.15, 0.30, 0, physicsMats.chassisMaterial, 0.06);
    this.leftShin.addShape(new CANNON.Box(new CANNON.Vec3(0.05, 0.21, 0.055)));

    this.rightShin = makeBody(1.6, -0.15, 0.30, 0, physicsMats.chassisMaterial, 0.06);
    this.rightShin.addShape(new CANNON.Box(new CANNON.Vec3(0.05, 0.21, 0.055)));

    // 6 & 7. High-friction Feet (flat sole + chamfered heel/toe prevents snagging without rocking)
    this.leftFoot = makeBody(1.0, 0.15, 0.06, 0.03, physicsMats.footMaterial, 0.12);
    this.leftFoot.addShape(new CANNON.Box(new CANNON.Vec3(0.068, 0.028, 0.068)));
    this.leftFoot.addShape(new CANNON.Sphere(0.027), new CANNON.Vec3(0, 0.0, -0.072));
    this.leftFoot.addShape(new CANNON.Sphere(0.027), new CANNON.Vec3(0, 0.0, 0.072));

    this.rightFoot = makeBody(1.0, -0.15, 0.06, 0.03, physicsMats.footMaterial, 0.12);
    this.rightFoot.addShape(new CANNON.Box(new CANNON.Vec3(0.068, 0.028, 0.068)));
    this.rightFoot.addShape(new CANNON.Sphere(0.027), new CANNON.Vec3(0, 0.0, -0.072));
    this.rightFoot.addShape(new CANNON.Sphere(0.027), new CANNON.Vec3(0, 0.0, 0.072));

    // 8 & 9. Balancer Arms
    this.leftArm = makeBody(0.9, 0.27, 1.22, 0, physicsMats.chassisMaterial, 0.08);
    this.leftArm.addShape(new CANNON.Box(new CANNON.Vec3(0.04, 0.22, 0.045)));

    this.rightArm = makeBody(0.9, -0.27, 1.22, 0, physicsMats.chassisMaterial, 0.08);
    this.rightArm.addShape(new CANNON.Box(new CANNON.Vec3(0.04, 0.22, 0.045)));

    this.allBodies = [
      this.torso,
      this.leftThigh,
      this.rightThigh,
      this.leftShin,
      this.rightShin,
      this.leftFoot,
      this.rightFoot,
      this.leftArm,
      this.rightArm,
    ];

    for (const b of this.allBodies) {
      this.world.addBody(b);
    }

    // Motorized Hinge Constraints
    const axisX = new CANNON.Vec3(1, 0, 0);

    this.leftHip = new CANNON.HingeConstraint(this.torso, this.leftThigh, {
      pivotA: new CANNON.Vec3(0.15, -0.29, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.22, 0),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });
    this.rightHip = new CANNON.HingeConstraint(this.torso, this.rightThigh, {
      pivotA: new CANNON.Vec3(-0.15, -0.29, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.22, 0),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });

    this.leftKnee = new CANNON.HingeConstraint(this.leftThigh, this.leftShin, {
      pivotA: new CANNON.Vec3(0, -0.22, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.21, 0),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });
    this.rightKnee = new CANNON.HingeConstraint(this.rightThigh, this.rightShin, {
      pivotA: new CANNON.Vec3(0, -0.22, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.21, 0),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });

    this.leftAnkle = new CANNON.HingeConstraint(this.leftShin, this.leftFoot, {
      pivotA: new CANNON.Vec3(0, -0.21, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.03, -0.03),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });
    this.rightAnkle = new CANNON.HingeConstraint(this.rightShin, this.rightFoot, {
      pivotA: new CANNON.Vec3(0, -0.21, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.03, -0.03),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });

    this.leftShoulder = new CANNON.HingeConstraint(this.torso, this.leftArm, {
      pivotA: new CANNON.Vec3(0.27, 0.20, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.22, 0),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });
    this.rightShoulder = new CANNON.HingeConstraint(this.torso, this.rightArm, {
      pivotA: new CANNON.Vec3(-0.27, 0.20, 0),
      axisA: axisX,
      pivotB: new CANNON.Vec3(0, 0.22, 0),
      axisB: axisX,
      collideConnected: false,
      maxForce: 1e6,
    });

    this.allConstraints = [
      this.leftHip,
      this.rightHip,
      this.leftKnee,
      this.rightKnee,
      this.leftAnkle,
      this.rightAnkle,
      this.leftShoulder,
      this.rightShoulder,
    ];

    for (const c of this.allConstraints) {
      c.enableMotor();
      c.setMotorSpeed(0);
      c.setMotorMaxForce(80);
      this.world.addConstraint(c);
    }

    this.leftHip.setMotorMaxForce(480);
    this.rightHip.setMotorMaxForce(480);
    this.leftKnee.setMotorMaxForce(360);
    this.rightKnee.setMotorMaxForce(360);
    this.leftAnkle.setMotorMaxForce(135);
    this.rightAnkle.setMotorMaxForce(135);
    this.leftShoulder.setMotorMaxForce(85);
    this.rightShoulder.setMotorMaxForce(85);

    // Register ground contact listeners for disqualification (torso/thighs/arms)
    this.attachContactListeners();

    if (!headless && assets) {
      this.buildCompositeVisuals(assets);
      this.syncVisuals();
    }
  }

  private attachContactListeners(): void {
    const markBodyContact = (reason: string) => (e: { body: CANNON.Body }) => {
      if (!this.alive) return;
      if (e.body === this.groundBody || (e.body.collisionFilterGroup & COLLISION_GROUP_WORLD) !== 0) {
        // Check if it was a floor or low obstacle contact
        this.hazardBodyContact = true;
        this.hazardReason = reason;
      }
    };

    this.torso.addEventListener('collide', markBodyContact('Касание пола торсом/головой'));
    this.leftThigh.addEventListener('collide', markBodyContact('Падение на левое колено/бедро'));
    this.rightThigh.addEventListener('collide', markBodyContact('Падение на правое колено/бедро'));
    this.leftArm.addEventListener('collide', markBodyContact('Касание пола рукой'));
    this.rightArm.addEventListener('collide', markBodyContact('Касание пола рукой'));
  }

  private buildCompositeVisuals(assets: ProceduralAssets): void {
    const titaniumMat = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      map: assets.brushedMetalTexture,
      roughness: 0.28,
      metalness: 0.82,
    });
    const carbonMat = new THREE.MeshStandardMaterial({
      color: 0x262f3d,
      map: assets.carbonTexture,
      roughness: 0.65,
      metalness: 0.24,
    });
    const chromeMat = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      roughness: 0.14,
      metalness: 0.95,
    });
    const rubberMat = new THREE.MeshStandardMaterial({
      color: 0x11151c,
      roughness: 0.92,
      metalness: 0.05,
    });
    this.accentMaterial = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      roughness: 0.35,
      metalness: 0.6,
    });
    this.coreGlowMaterial = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 1.4,
      roughness: 0.2,
      metalness: 0.3,
    });

    this.clonedMaterials.push(
      titaniumMat,
      carbonMat,
      chromeMat,
      rubberMat,
      this.accentMaterial,
      this.coreGlowMaterial,
    );

    const regShadow = (obj: THREE.Object3D) => {
      obj.traverse((c) => {
        if ((c as THREE.Mesh).isMesh) {
          c.castShadow = true;
          c.receiveShadow = true;
        }
      });
    };

    // 1. Torso + Head Composite Group (9 parts)
    const torsoGroup = new THREE.Group();
    const pelvisMesh = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.16, 0.20), carbonMat);
    pelvisMesh.position.set(0, -0.24, 0);
    torsoGroup.add(pelvisMesh);

    const spineMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.095, 0.34, 12), carbonMat);
    spineMesh.position.set(0, -0.05, -0.02);
    torsoGroup.add(spineMesh);

    // Sculpted extruded chest shell
    const chestShape = new THREE.Shape();
    chestShape.moveTo(-0.18, -0.15);
    chestShape.lineTo(0.18, -0.15);
    chestShape.lineTo(0.21, 0.08);
    chestShape.lineTo(0.15, 0.19);
    chestShape.lineTo(-0.15, 0.19);
    chestShape.lineTo(-0.21, 0.08);
    chestShape.closePath();
    const chestGeo = new THREE.ExtrudeGeometry(chestShape, {
      depth: 0.21,
      bevelEnabled: true,
      bevelSize: 0.022,
      bevelThickness: 0.022,
      bevelSegments: 2,
    });
    chestGeo.center();
    const chestMesh = new THREE.Mesh(chestGeo, titaniumMat);
    chestMesh.position.set(0, 0.06, 0.01);
    torsoGroup.add(chestMesh);

    // Gyroscope core ring
    const gyroRing = new THREE.Mesh(new THREE.TorusGeometry(0.075, 0.018, 10, 24), this.coreGlowMaterial);
    gyroRing.position.set(0, 0.06, 0.14);
    torsoGroup.add(gyroRing);

    const gyroInner = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.03, 16), chromeMat);
    gyroInner.rotation.x = Math.PI / 2;
    gyroInner.position.set(0, 0.06, 0.13);
    torsoGroup.add(gyroInner);

    // Hip & Shoulder Servo Drums
    for (const side of [-1, 1]) {
      const hipDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.09, 14), this.accentMaterial);
      hipDrum.rotation.z = Math.PI / 2;
      hipDrum.position.set(side * 0.15, -0.29, 0);
      torsoGroup.add(hipDrum);

      const shoulderDrum = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.08, 14), this.accentMaterial);
      shoulderDrum.rotation.z = Math.PI / 2;
      shoulderDrum.position.set(side * 0.24, 0.20, 0);
      torsoGroup.add(shoulderDrum);
    }

    // Neck & Optical Head Pod
    const neckMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.055, 0.12, 10), chromeMat);
    neckMesh.position.set(0, 0.27, 0.01);
    torsoGroup.add(neckMesh);

    const headShell = new THREE.Mesh(new THREE.BoxGeometry(0.19, 0.14, 0.20), titaniumMat);
    headShell.position.set(0, 0.37, 0.02);
    torsoGroup.add(headShell);

    const visorStrip = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.055, 0.03), this.coreGlowMaterial);
    visorStrip.position.set(0, 0.375, 0.115);
    torsoGroup.add(visorStrip);

    regShadow(torsoGroup);
    this.rootGroup.add(torsoGroup);
    this.bodyMeshes.push({ body: this.torso, group: torsoGroup });

    // 2 & 3. Thigh Composite Groups (4 parts each)
    const buildThigh = (body: CANNON.Body) => {
      const g = new THREE.Group();
      const core = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.42, 0.11), carbonMat);
      g.add(core);
      const plate = new THREE.Mesh(new THREE.BoxGeometry(0.115, 0.30, 0.07), titaniumMat);
      plate.position.set(0, 0.03, 0.045);
      g.add(plate);
      const piston = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.34, 8), chromeMat);
      piston.position.set(0, 0, -0.065);
      g.add(piston);
      const kneeHub = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.13, 14), this.accentMaterial);
      kneeHub.rotation.z = Math.PI / 2;
      kneeHub.position.set(0, -0.22, 0);
      g.add(kneeHub);

      regShadow(g);
      this.rootGroup.add(g);
      this.bodyMeshes.push({ body, group: g });
    };
    buildThigh(this.leftThigh);
    buildThigh(this.rightThigh);

    // 4 & 5. Shin Composite Groups (4 parts each)
    const buildShin = (body: CANNON.Body) => {
      const g = new THREE.Group();
      const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.032, 0.40, 10), carbonMat);
      g.add(strut);
      const guard = new THREE.Mesh(new THREE.BoxGeometry(0.095, 0.30, 0.055), titaniumMat);
      guard.position.set(0, 0.03, 0.042);
      g.add(guard);
      const damper = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.30, 8), chromeMat);
      damper.position.set(0, 0.01, -0.052);
      g.add(damper);
      const ankleHub = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.11, 12), this.accentMaterial);
      ankleHub.rotation.z = Math.PI / 2;
      ankleHub.position.set(0, -0.21, 0);
      g.add(ankleHub);

      regShadow(g);
      this.rootGroup.add(g);
      this.bodyMeshes.push({ body, group: g });
    };
    buildShin(this.leftShin);
    buildShin(this.rightShin);

    // 6 & 7. Foot Composite Groups (3 parts each)
    const buildFoot = (body: CANNON.Body) => {
      const g = new THREE.Group();
      const bridge = new THREE.Mesh(new THREE.BoxGeometry(0.115, 0.035, 0.21), titaniumMat);
      bridge.position.set(0, 0.008, 0);
      g.add(bridge);
      const sole = new THREE.Mesh(new THREE.BoxGeometry(0.135, 0.024, 0.238), rubberMat);
      sole.position.set(0, -0.018, 0);
      g.add(sole);
      const toeCap = new THREE.Mesh(new THREE.BoxGeometry(0.125, 0.036, 0.05), this.accentMaterial);
      toeCap.position.set(0, -0.005, 0.098);
      g.add(toeCap);

      regShadow(g);
      this.rootGroup.add(g);
      this.bodyMeshes.push({ body, group: g });
    };
    buildFoot(this.leftFoot);
    buildFoot(this.rightFoot);

    // 8 & 9. Balancer Arm Groups (3 parts each)
    const buildArm = (body: CANNON.Body) => {
      const g = new THREE.Group();
      const link = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.024, 0.42, 10), carbonMat);
      g.add(link);
      const elbow = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.16, 0.075), titaniumMat);
      elbow.position.set(0, 0.06, 0);
      g.add(elbow);
      const counterweight = new THREE.Mesh(new THREE.SphereGeometry(0.048, 12, 10), this.accentMaterial);
      counterweight.position.set(0, -0.21, 0);
      g.add(counterweight);

      regShadow(g);
      this.rootGroup.add(g);
      this.bodyMeshes.push({ body, group: g });
    };
    buildArm(this.leftArm);
    buildArm(this.rightArm);

    // Contact Ground Shadow Disc directly beneath torso at y = 0.015
    const shadowGeo = new THREE.PlaneGeometry(1.1, 1.1);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      map: assets.contactShadowTexture,
      transparent: true,
      depthWrite: false,
      opacity: 0.48,
    });
    this.contactShadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
    this.contactShadowMesh.position.set(0, 0.015, 0);
    this.rootGroup.add(this.contactShadowMesh);
  }

  public setHighlighted(highlighted: boolean): void {
    if (this.isHighlighted === highlighted || this.ghosted) return;
    this.isHighlighted = highlighted;
    if (this.coreGlowMaterial) {
      this.coreGlowMaterial.color.setHex(highlighted ? 0xf59e0b : 0x38bdf8);
      this.coreGlowMaterial.emissive.setHex(highlighted ? 0xd97706 : 0x0284c7);
      this.coreGlowMaterial.emissiveIntensity = highlighted ? 2.1 : 1.1;
    }
    if (this.accentMaterial) {
      this.accentMaterial.color.setHex(highlighted ? 0xf59e0b : 0x64748b);
    }
  }

  public updateFootContacts(world: CANNON.World): void {
    this.leftFootContact = false;
    this.rightFootContact = false;
    for (let i = 0; i < world.contacts.length; i++) {
      const c = world.contacts[i];
      if (c.bi === this.leftFoot || c.bj === this.leftFoot) {
        this.leftFootContact = true;
      }
      if (c.bi === this.rightFoot || c.bj === this.rightFoot) {
        this.rightFootContact = true;
      }
    }
  }

  public stepControl(dt: number, headwindForce: number): void {
    if (!this.alive) return;

    this.age += dt;

    // 1. Check disqualification rules
    if (this.hazardBodyContact) {
      this.disqualify(this.hazardReason);
      return;
    }

    const torsoY = this.torso.position.y;
    const leftKneeY = (this.leftThigh.position.y + this.leftShin.position.y) * 0.5;
    const rightKneeY = (this.rightThigh.position.y + this.rightShin.position.y) * 0.5;

    // Torso Euler angles
    _threeQuat.set(
      this.torso.quaternion.x,
      this.torso.quaternion.y,
      this.torso.quaternion.z,
      this.torso.quaternion.w,
    );
    _euler.setFromQuaternion(_threeQuat, 'XYZ');
    const pitch = _euler.x;
    const yaw = _euler.y;
    const roll = _euler.z;

    if (torsoY < 0.72) {
      this.disqualify('Падение центра масс ниже порога');
      return;
    }
    if (leftKneeY < 0.21 || rightKneeY < 0.21) {
      this.disqualify('Колено коснулось дорожки');
      return;
    }
    if (Math.abs(pitch) > 1.25 || Math.abs(roll) > 1.25) {
      this.disqualify('Критическое опрокидывание корпуса');
      return;
    }
    if (Math.abs(this.torso.position.x) > 4.2) {
      this.disqualify('Выход за пределы дорожки');
      return;
    }

    // 2. Compute joint flexion angles
    const lHipAngle = computeHingeAngleX(this.torso, this.leftThigh);
    const rHipAngle = computeHingeAngleX(this.torso, this.rightThigh);
    const lKneeAngle = computeHingeAngleX(this.leftThigh, this.leftShin);
    const rKneeAngle = computeHingeAngleX(this.rightThigh, this.rightShin);
    const lAnkleAngle = computeHingeAngleX(this.leftShin, this.leftFoot);
    const rAnkleAngle = computeHingeAngleX(this.rightShin, this.rightFoot);
    const lArmAngle = computeHingeAngleX(this.torso, this.leftArm);
    const rArmAngle = computeHingeAngleX(this.torso, this.rightArm);

    // 3. Populate 18 sensory inputs
    const s = this.sensorBuffer;
    s[0] = THREE.MathUtils.clamp(pitch / (Math.PI * 0.5), -1, 1);
    s[1] = THREE.MathUtils.clamp(yaw / (Math.PI * 0.5), -1, 1);
    s[2] = THREE.MathUtils.clamp(roll / (Math.PI * 0.5), -1, 1);

    s[3] = THREE.MathUtils.clamp(this.torso.angularVelocity.x / 6, -1, 1);
    s[4] = THREE.MathUtils.clamp(this.torso.angularVelocity.y / 6, -1, 1);
    s[5] = THREE.MathUtils.clamp(this.torso.angularVelocity.z / 6, -1, 1);

    s[6] = THREE.MathUtils.clamp(this.torso.velocity.x / 4, -1, 1);
    s[7] = THREE.MathUtils.clamp(this.torso.velocity.y / 4, -1, 1);
    s[8] = THREE.MathUtils.clamp(this.torso.velocity.z / 5, -1, 1);

    s[9] = THREE.MathUtils.clamp(lHipAngle / 1.4, -1, 1);
    s[10] = THREE.MathUtils.clamp(rHipAngle / 1.4, -1, 1);
    s[11] = THREE.MathUtils.clamp((lKneeAngle - 0.7) / 1.2, -1, 1);
    s[12] = THREE.MathUtils.clamp((rKneeAngle - 0.7) / 1.2, -1, 1);

    s[13] = this.leftFootContact ? 1 : 0;
    s[14] = this.rightFootContact ? 1 : 0;

    const phase = 2 * Math.PI * this.cpgFreq * this.age;
    s[15] = Math.sin(phase);
    s[16] = Math.cos(phase);
    s[17] = THREE.MathUtils.clamp((torsoY - 0.7) / 0.7, -1, 1);

    // 4. Forward pass through Neural Network -> 6 motor speed commands
    const out = this.brain.forward(s);

    const maxHipSpeed = 6.5;
    const maxKneeSpeed = 8.0;
    const maxArmSpeed = 5.5;

    // Soft anatomical limit enforcement (note: in cannon-es, positive motorSpeed rotates bodyB by -angle around axisA)
    const clampJointMotor = (
      angle: number,
      desiredSpeed: number,
      minAngle: number,
      maxAngle: number,
    ): number => {
      if (angle < minAngle && desiredSpeed > 0) return (angle - minAngle) * 6.0; // negative speed increases angle
      if (angle > maxAngle && desiredSpeed < 0) return (angle - maxAngle) * 6.0; // positive speed decreases angle
      return desiredSpeed;
    };

    this.leftHip.setMotorSpeed(clampJointMotor(lHipAngle, out[0] * maxHipSpeed, -1.2, 0.95));
    this.rightHip.setMotorSpeed(clampJointMotor(rHipAngle, out[1] * maxHipSpeed, -1.2, 0.95));
    this.leftKnee.setMotorSpeed(clampJointMotor(lKneeAngle, out[2] * maxKneeSpeed, 0.02, 2.05));
    this.rightKnee.setMotorSpeed(clampJointMotor(rKneeAngle, out[3] * maxKneeSpeed, 0.02, 2.05));
    this.leftShoulder.setMotorSpeed(clampJointMotor(lArmAngle, out[4] * maxArmSpeed, -1.15, 1.15));
    this.rightShoulder.setMotorSpeed(clampJointMotor(rArmAngle, out[5] * maxArmSpeed, -1.15, 1.15));

    // Compliant world-level ankle spring-damper (stance ankle resists toppling + swing dorsiflexes toe up)
    const shinWorldAngleL = pitch + lHipAngle + lKneeAngle;
    const shinWorldAngleR = pitch + rHipAngle + rKneeAngle;
    const ankleBalance = THREE.MathUtils.clamp(pitch * 0.45 + this.torso.angularVelocity.x * 0.08, -0.35, 0.35);
    const targetAnkleL = this.leftFootContact ? -shinWorldAngleL + ankleBalance : -0.98 * shinWorldAngleL - 0.18;
    const targetAnkleR = this.rightFootContact ? -shinWorldAngleR + ankleBalance : -0.98 * shinWorldAngleR - 0.18;
    this.leftAnkle.setMotorSpeed(THREE.MathUtils.clamp((lAnkleAngle - targetAnkleL) * 9.5, -7, 7));
    this.rightAnkle.setMotorSpeed(THREE.MathUtils.clamp((rAnkleAngle - targetAnkleR) * 9.5, -7, 7));

    // 5. Realistic capped hip-abductor / lateral balance stabilization (3D roll & yaw compliance)
    // Capped so heavy side impacts, bad gaits, or obstacles still cause sway and falls
    const rollTorque = THREE.MathUtils.clamp(-roll * 175 - this.torso.angularVelocity.z * 36, -110, 110);
    const yawTorque = THREE.MathUtils.clamp(-yaw * 115 - this.torso.angularVelocity.y * 26, -75, 75);
    this.torso.torque.z += rollTorque;
    this.torso.torque.y += yawTorque;

    // 6. Apply headwind if challenge active
    if (headwindForce > 0) {
      const gust = headwindForce * (0.85 + 0.3 * Math.sin(this.age * 3.1 + this.id));
      this.torso.applyForce(new CANNON.Vec3(0, 0, -gust), this.torso.position);
    }

    // 6. Strict Fitness Function for Bipedal Walking
    this.distance = this.torso.position.z;
    if (this.distance > this.maxDistance) {
      this.maxDistance = this.distance;
    }

    // Upright posture bonus
    if (torsoY > 1.08) {
      this.uprightAccumulator += dt * (torsoY - 0.9) * 3.5;
    }

    // Stability penalty (lateral X drift + roll/yaw sway)
    const lateralDrift = Math.abs(this.torso.position.x);
    const sway = Math.abs(roll) * 2.2 + Math.abs(yaw) * 1.4 + Math.max(0, Math.abs(pitch) - 0.28) * 2.0;
    this.stabilityPenalty += dt * (lateralDrift * 1.8 + sway);

    // Combined fitness: strongly reward forward +Z progress + upright posture - instability
    this.fitness =
      this.maxDistance * 14.0 +
      Math.max(0, this.distance) * 4.0 +
      this.uprightAccumulator -
      this.stabilityPenalty;
  }

  public disqualify(reason: string): void {
    if (!this.alive) return;
    this.alive = false;
    this.disqualifiedReason = reason;

    // Freeze all physics bodies and disable collisions so ghosts remain in place
    for (const b of this.allBodies) {
      b.velocity.set(0, 0, 0);
      b.angularVelocity.set(0, 0, 0);
      b.type = CANNON.Body.STATIC;
      b.mass = 0;
      b.updateMassProperties();
      b.collisionFilterGroup = 0;
      b.collisionFilterMask = 0;
    }

    for (const c of this.allConstraints) {
      c.disableMotor();
    }

    // Make 3D meshes semi-transparent
    if (!this.ghosted) {
      this.ghosted = true;
      for (const mat of this.clonedMaterials) {
        mat.transparent = true;
        mat.opacity = 0.24;
        mat.depthWrite = false;
      }
      if (this.coreGlowMaterial) {
        this.coreGlowMaterial.emissiveIntensity = 0.15;
        this.coreGlowMaterial.color.setHex(0x64748b);
      }
      if (this.contactShadowMesh) {
        this.contactShadowMesh.visible = false;
      }
    }
  }

  public syncVisuals(): void {
    for (let i = 0; i < this.bodyMeshes.length; i++) {
      const { body, group } = this.bodyMeshes[i];
      group.position.set(body.position.x, body.position.y, body.position.z);
      group.quaternion.set(body.quaternion.x, body.quaternion.y, body.quaternion.z, body.quaternion.w);
    }
    if (this.contactShadowMesh && this.alive) {
      const heightFactor = THREE.MathUtils.clamp(1.3 - (this.torso.position.y - 1.1) * 0.5, 0.5, 1.1);
      this.contactShadowMesh.position.set(this.torso.position.x, 0.015, this.torso.position.z);
      this.contactShadowMesh.scale.set(heightFactor, 1, heightFactor);
    }
  }

  public getSnapshot(): AgentSnapshot {
    _threeQuat.set(
      this.torso.quaternion.x,
      this.torso.quaternion.y,
      this.torso.quaternion.z,
      this.torso.quaternion.w,
    );
    _euler.setFromQuaternion(_threeQuat, 'XYZ');

    return {
      id: this.id,
      alive: this.alive,
      distance: this.distance,
      maxDistance: this.maxDistance,
      fitness: this.fitness,
      speed: this.torso.velocity.z,
      torsoHeight: this.torso.position.y,
      pitchDeg: THREE.MathUtils.radToDeg(_euler.x),
      rollDeg: THREE.MathUtils.radToDeg(_euler.z),
      leftContact: this.leftFootContact,
      rightContact: this.rightFootContact,
      disqualifiedReason: this.disqualifiedReason || undefined,
    };
  }

  public dispose(scene: THREE.Scene): void {
    for (const c of this.allConstraints) {
      this.world.removeConstraint(c);
    }
    for (const b of this.allBodies) {
      this.world.removeBody(b);
    }
    scene.remove(this.rootGroup);
    this.rootGroup.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) {
        (obj as THREE.Mesh).geometry.dispose();
      }
    });
    for (const m of this.clonedMaterials) {
      m.dispose();
    }
    if (this.contactShadowMesh) {
      (this.contactShadowMesh.material as THREE.Material).dispose();
    }
  }
}

/**
 * Generates a DAgger-distilled CPG + vestibular Capture-Point neural network (18 -> 16 -> 6)
 * verified to walk 31.0m+ in 28s through pure physics motor commands.
 */
export function createPretrainedBipedNetwork(): NeuralNetwork {
  return NeuralNetwork.deserialize({
    inputSize: 18,
    hiddenSize: 16,
    outputSize: 6,
    activationFn: 'tanh',
    wIH: [
      0.4427, -0.0385, -0.0412, 0.9533, 0.028, 0.1546, 0.0592, 0.3007, 0.641,
      -0.6876, -0.987, -0.5071, -0.015, 0.0647, 0.2917, -1.3864, 0.1952, -0.4793,
      -0.5429, 0.1619, -0.1635, 0.5962, -0.2381, -0.2479, -0.0517, 0.1618, -0.2715,
      -0.5958, -1.0137, 0.3064, -0.0409, 0.5098, -0.3073, 0.143, -0.655, 0.0266,
      -1.6877, -0.1817, -0.3037, -0.8092, -0.0532, 0.2514, -0.341, -0.474, -0.9208,
      -0.2718, 0.3265, 0.0035, 0.5684, 0.5805, -0.0686, -0.8776, 0.1127, 0.9636,
      0.0201, 0.0811, -0.1088, -0.0864, 0.2528, 0.0542, 0.0502, 0.2329, 0.0444,
      -0.0256, 0.059, -0.251, 0.2424, -0.0439, -0.2299, -0.5392, -1.763, -0.0256,
      0.0166, 0.053, 0.0002, 0.0474, 0.0535, 0.227, 0.2076, 0.0205, -0.09,
      0.2925, -0.2891, -0.2385, -0.0645, -0.3581, -0.0794, -0.5015, 0.9187, 0.2349,
      0.2152, -0.2919, 0.0466, 0.0097, -0.0281, -0.0402, 0.0168, 0.1481, 0.0889,
      -0.2222, -0.1382, -0.1588, 1.2225, -0.0218, -0.2528, -0.4259, 0.0177, -0.1919,
      0.1464, 0.0084, -0.007, -0.1369, -0.0311, 0.0164, 0.0368, 0.1756, -0.2934,
      -0.0271, -0.1253, -0.0636, -0.2477, -0.1788, 0.067, 0.5637, 1.674, -0.1297,
      1.2586, 0.0109, -0.1365, 0.2454, 0.0371, -0.417, -0.0729, 0.0537, 0.4545,
      -0.1065, -0.3521, 0.2128, -0.2839, 0.2067, 1.0085, 0.149, 0.0812, -0.3819,
      -1.0173, 0.0599, -0.186, -0.1767, 0.2077, -0.0606, -0.1406, -0.1879, -0.412,
      -0.4352, 0.7911, -0.1905, -0.6091, -0.6441, -0.1912, 0.1196, -0.2343, 0.1906,
      0.2597, -0.0014, 0.0404, -0.0157, -0.0782, 0.119, 0.0643, 0.3187, 0.1418,
      0.3332, -0.456, -1.0954, -0.0977, 0.5147, 0.0593, -0.548, -0.2582, 0.0477,
      0.2923, 0.1591, -0.1367, -0.1401, 0.0102, 0.1555, 0.0427, 0.5613, 0.7303,
      -0.3527, -0.2252, 0.5438, -0.8216, -0.3586, -0.2396, 1.2809, -0.4504, -0.0597,
      0.3061, 0.3953, -0.0678, 0.1425, -0.0144, 0.1702, 0.0035, -0.1833, 0.0433,
      -0.0152, 0.1778, 0.221, -0.4891, -0.0441, -0.2905, -1.0195, -0.425, 0.0619,
      -0.3494, -0.1459, -0.0044, 0.6303, 0.3515, 0.0142, -0.0635, -0.1459, -0.3518,
      -0.7868, -0.5854, -0.1199, 0.1291, -0.5063, 0.5076, -0.1525, 0.6454, 0.0182,
      -1.3016, -0.1281, 0.1892, -0.8324, -0.0824, 0.3549, 0.3396, -0.4188, -0.7935,
      0.5937, -0.5691, 0.2781, 0.014, -0.0684, 0.7109, 0.7739, -0.1102, 0.7434,
      -0.3808, -0.0612, -0.0491, -0.3249, -0.0787, -0.0122, -0.1175, 0.0384, -0.1782,
      -0.3076, 0.1174, 0.6593, 0.0973, 0.0642, 0.5005, -0.4474, -0.3172, -0.086,
      -0.7401, -0.0063, 0.1719, -0.1653, 0.4146, 0.4045, 0.1131, -0.1685, -0.2848,
      0.7546, -0.1867, 0.5954, 0.4746, 0.1558, -0.5696, -0.2437, -0.2289, 0.1715,
    ],
    bH: [
      -0.3642, -0.1835, 0.519, -0.5162, 0.0525, -0.026, -0.5349, -0.0286,
      -0.0132, -0.0081, 0.075, 0.2055, -0.1236, 0.2813, 0.12, 0.0481,
    ],
    wHO: [
      -0.8808, -0.7968, -2.3887, 0.0477, 0.641, -0.4371, 0.1149, 0.224,
      -1.3736, 0.4242, -0.7791, 0.5218, -1.1378, 1.1814, -0.619, 0.7608,
      -1.1477, -1.3434, 0.822, 0.4091, -0.5981, 0.2541, -0.1575, 1.3634,
      0.3542, -0.5556, -1.0979, 0.2088, -0.8874, -2.1933, -0.0268, -0.9291,
      -0.0501, 0.2281, -0.2309, -0.6832, -0.2202, -0.1983, -1.0024, 0.2351,
      -0.2693, -1.1479, 0.2091, 0.6065, -0.138, -0.1322, 0.8852, 0.591,
      -0.1206, -0.2463, 0.0394, -1.0941, 0.0811, 1.2095, -0.6644, -0.0442,
      -0.2608, -0.3086, -0.2746, -0.7744, 0.1483, -0.1545, 0.063, 0.3698,
      -0.0073, -0.0807, 0.0628, 0.1578, 0.566, 0.1607, -0.2167, -0.0282,
      -0.0874, 0.1, -0.0776, 0.2691, 0.0519, -0.0484, 0.2302, -0.0331,
      0.0073, 0.0807, -0.0628, -0.1578, -0.566, -0.1607, 0.2167, 0.0282,
      0.0874, -0.1, 0.0776, -0.2691, -0.0519, 0.0484, -0.2302, 0.0331,
    ],
    bO: [0.3477, 0.2471, -0.175, 0.1225, -0.1065, 0.1065],
  });
}
