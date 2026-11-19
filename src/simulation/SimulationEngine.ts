import * as CANNON from 'cannon-es';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { NeuralNetwork } from '../ai/NeuralNetwork';
import {
  BIPED_INPUT_LABELS,
  BIPED_OUTPUT_LABELS,
  BipedWalker3D,
  COLLISION_GROUP_PROJECTILE,
  COLLISION_GROUP_ROBOT,
  COLLISION_GROUP_WORLD,
  createPretrainedBipedNetwork,
} from './BipedWalker3D';
import {
  DRONE_INPUT_LABELS,
  DRONE_OUTPUT_LABELS,
  DroneAgent3D,
  createPretrainedDroneNetwork,
} from './DroneAgent3D';
import {
  MANUAL_CAR_INPUT_LABELS,
  MANUAL_CAR_OUTPUT_LABELS,
  ManualCarAgent3D,
  createManualCarCircuitData,
  createPretrainedManualCarNetwork,
  type CircuitData,
} from './ManualCarAgent3D';
import {
  createDistanceMarkerTexture,
  createProceduralAssets,
  type ProceduralAssets,
} from './materials';
import {
  BALANCE_INPUT_LABELS,
  BALANCE_OUTPUT_LABELS,
  CartPoleAgent3D,
  PARKOUR_INPUT_LABELS,
  PARKOUR_OUTPUT_LABELS,
  ParkourAgent3D,
  ROVER_INPUT_LABELS,
  ROVER_OUTPUT_LABELS,
  RoverAgent3D,
  createPretrainedCartPoleNetwork,
  type SimpleModeAgent,
} from './OtherModes';
import type {
  AgentSnapshot,
  EnvironmentChallenges,
  EvolutionHyperparams,
  GenerationHistoryPoint,
  GenomeData,
  NetworkInspectorData,
  PersistedModeState,
  SimulationMode,
  SimulationTelemetry,
} from './types';

interface ProjectileItem {
  body: CANNON.Body;
  mesh: THREE.Mesh;
  age: number;
}

interface TerrainObstacle {
  body: CANNON.Body;
  mesh: THREE.Mesh;
}

export interface PersistedGenerationPayload {
  mode: SimulationMode;
  generation: number;
  bestDistanceEver: number;
  bestFitnessEver: number;
  milestone30mReached: boolean;
  hyperparams: EvolutionHyperparams;
  bestGenome: GenomeData;
  history: GenerationHistoryPoint[];
}

const _camDelta = new THREE.Vector3();

export class SimulationEngine {
  private readonly container: HTMLElement;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly assets: ProceduralAssets;
  private readonly isSoftwareGl: boolean;

  private keyLight!: THREE.DirectionalLight;
  private envGroup = new THREE.Group();
  private challengeGroup = new THREE.Group();
  private windGroup = new THREE.Group();
  private windStreaks!: THREE.InstancedMesh;
  private windPositions: Array<{ x: number; y: number; z: number; speed: number }> = [];

  private world!: CANNON.World;
  private groundBody!: CANNON.Body;
  private groundMat!: CANNON.Material;
  private chassisMat!: CANNON.Material;
  private footMat!: CANNON.Material;

  // State
  private mode: SimulationMode = 'biomechanics';
  private paused = false;
  private simSpeed = 1;
  private followLeader = true;
  private userOrbiting = false;
  private generation = 1;
  private elapsedTime = 0;
  private generationDuration = 28;
  private bestDistanceEver = 0;
  private bestFitnessEver = 0;
  private milestone30mReached = false;
  private history: GenerationHistoryPoint[] = [];
  private bestGenomeEver: GenomeData | null = null;

  private hyperparams: EvolutionHyperparams = {
    populationSize: 12,
    mutationRate: 0.16,
    mutationStrength: 0.24,
    activationFn: 'tanh',
    elitismCount: 2,
    hallOfFame: true,
  };

  private challenges: EnvironmentChallenges = {
    unevenTerrain: false,
    headwind: false,
    windForce: 22,
    ballsThrown: 0,
    manualUnlock30m: false,
  };

  // Populations
  private bipedAgents: BipedWalker3D[] = [];
  private simpleAgents: SimpleModeAgent[] = [];
  private trackedId = 0;

  // Dynamic environment objects
  private terrainObstacles: TerrainObstacle[] = [];
  private projectiles: ProjectileItem[] = [];
  private roverObstacles: Array<{ x: number; z: number; r: number }> = [];
  private roverBeacons: THREE.Vector3[] = [];
  private parkourHurdles: Array<{ z: number; h: number }> = [];
  private carCircuit: CircuitData = createManualCarCircuitData();
  private droneCheckpoints: THREE.Vector3[] = [];

  // Loop & callbacks
  private frameId = 0;
  private lastPerfTime = performance.now();
  private telemetryTimer = 0;
  private disposed = false;
  private readonly onTelemetry: (data: SimulationTelemetry) => void;
  private readonly onContextLostCallback?: (lost: boolean) => void;
  private readonly onGenerationComplete?: (payload: PersistedGenerationPayload) => void;

  constructor(
    container: HTMLElement,
    onTelemetry: (data: SimulationTelemetry) => void,
    onContextLost?: (lost: boolean) => void,
    onGenerationComplete?: (payload: PersistedGenerationPayload) => void,
  ) {
    this.container = container;
    this.onTelemetry = onTelemetry;
    this.onContextLostCallback = onContextLost;
    this.onGenerationComplete = onGenerationComplete;

    // Detect software WebGL
    this.isSoftwareGl = (() => {
      const gl = document.createElement('canvas').getContext('webgl2');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    this.renderer = new THREE.WebGLRenderer({
      antialias: !this.isSoftwareGl,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.setPixelRatio(this.isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = !this.isSoftwareGl;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setSize(container.clientWidth || window.innerWidth, container.clientHeight || window.innerHeight);

    container.appendChild(this.renderer.domElement);

    this.renderer.domElement.addEventListener('webglcontextlost', this.handleContextLost);
    this.renderer.domElement.addEventListener('webglcontextrestored', this.handleContextRestored);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x090d14);
    this.scene.fog = new THREE.FogExp2(0x090d14, 0.012);

    const pmremGenerator = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    pmremGenerator.dispose();

    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(48, w / h, 0.1, 320);
    this.camera.position.set(4.8, 2.7, -3.2);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = !this.isSoftwareGl;
    this.controls.dampingFactor = 0.06;
    this.controls.maxPolarAngle = Math.PI * 0.485;
    this.controls.minDistance = 2.0;
    this.controls.maxDistance = 48;
    this.controls.target.set(0, 1.0, 0.8);

    this.controls.addEventListener('start', () => {
      this.userOrbiting = true;
    });
    this.controls.addEventListener('end', () => {
      this.userOrbiting = false;
    });

    this.assets = createProceduralAssets(this.renderer.capabilities.getMaxAnisotropy());

    this.setupLighting();
    this.scene.add(this.envGroup);
    this.scene.add(this.challengeGroup);
    this.scene.add(this.windGroup);
    this.setupWindParticles();

    this.initMode(this.mode);

    if (!this.isSoftwareGl) {
      this.renderer.compile(this.scene, this.camera);
    }

    window.addEventListener('resize', this.handleResize);
    this.lastPerfTime = performance.now();
    this.startLoop();
  }

  private handleContextLost = (e: Event) => {
    e.preventDefault();
    this.onContextLostCallback?.(true);
  };

  private handleContextRestored = () => {
    this.onContextLostCallback?.(false);
  };

  private setupLighting(): void {
    const hemi = new THREE.HemisphereLight(0x93c5fd, 0x0f172a, 0.85);
    this.scene.add(hemi);

    this.keyLight = new THREE.DirectionalLight(0xfff1dc, 2.2);
    this.keyLight.position.set(12, 18, 8);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.width = 2048;
    this.keyLight.shadow.mapSize.height = 2048;
    const d = 18;
    this.keyLight.shadow.camera.left = -d;
    this.keyLight.shadow.camera.right = d;
    this.keyLight.shadow.camera.top = d;
    this.keyLight.shadow.camera.bottom = -d;
    this.keyLight.shadow.camera.near = 1;
    this.keyLight.shadow.camera.far = 60;
    this.keyLight.shadow.bias = -0.0005;
    this.keyLight.shadow.normalBias = 0.02;
    this.scene.add(this.keyLight);
    this.scene.add(this.keyLight.target);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.9);
    rimLight.position.set(-10, 7, -12);
    this.scene.add(rimLight);
  }

  private setupWindParticles(): void {
    const count = 80;
    const geo = new THREE.BoxGeometry(0.015, 0.015, 0.75);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x7dd3fc,
      transparent: true,
      opacity: 0.45,
    });
    this.windStreaks = new THREE.InstancedMesh(geo, mat, count);
    this.windStreaks.frustumCulled = false;
    this.windPositions = [];
    for (let i = 0; i < count; i++) {
      this.windPositions.push({
        x: (Math.random() - 0.5) * 6.5,
        y: 0.4 + Math.random() * 2.2,
        z: (Math.random() - 0.5) * 16,
        speed: 9 + Math.random() * 7,
      });
    }
    this.windGroup.add(this.windStreaks);
    this.windGroup.visible = false;
  }

  private initPhysicsWorld(): void {
    this.world = new CANNON.World({
      gravity: new CANNON.Vec3(0, -9.81, 0),
    });
    (this.world.solver as CANNON.GSSolver).iterations = 20;

    this.groundMat = new CANNON.Material('ground');
    this.chassisMat = new CANNON.Material('chassis');
    this.footMat = new CANNON.Material('foot');

    this.world.addContactMaterial(
      new CANNON.ContactMaterial(this.footMat, this.groundMat, {
        friction: 1.5,
        restitution: 0.0,
        contactEquationStiffness: 1e7,
        contactEquationRelaxation: 3,
      }),
    );
    this.world.addContactMaterial(
      new CANNON.ContactMaterial(this.chassisMat, this.groundMat, {
        friction: 0.45,
        restitution: 0.05,
      }),
    );

    this.groundBody = new CANNON.Body({
      mass: 0,
      material: this.groundMat,
      collisionFilterGroup: COLLISION_GROUP_WORLD,
      collisionFilterMask: COLLISION_GROUP_ROBOT | COLLISION_GROUP_PROJECTILE,
    });
    this.groundBody.addShape(new CANNON.Plane());
    this.groundBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
    this.world.addBody(this.groundBody);
  }

  private clearEnvironmentMeshes(): void {
    const clearGroup = (g: THREE.Group) => {
      while (g.children.length > 0) {
        const child = g.children[0];
        g.remove(child);
        child.traverse((o) => {
          if ((o as THREE.Mesh).isMesh) {
            (o as THREE.Mesh).geometry.dispose();
            const mat = (o as THREE.Mesh).material;
            if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
            else if (mat) mat.dispose();
          }
        });
      }
    };
    clearGroup(this.envGroup);
    clearGroup(this.challengeGroup);
    this.terrainObstacles = [];
    for (const p of this.projectiles) {
      this.scene.remove(p.mesh);
      p.mesh.geometry.dispose();
      (p.mesh.material as THREE.Material).dispose();
    }
    this.projectiles = [];
  }

  private clearPopulation(): void {
    for (const a of this.bipedAgents) a.dispose(this.scene);
    for (const a of this.simpleAgents) a.dispose(this.scene);
    this.bipedAgents = [];
    this.simpleAgents = [];
  }

  public setMode(newMode: SimulationMode): void {
    if (this.mode === newMode) return;
    this.mode = newMode;
    this.initMode(newMode);
  }

  public setHyperparams(patch: Partial<EvolutionHyperparams>): void {
    this.hyperparams = { ...this.hyperparams, ...patch };
    if (patch.activationFn) {
      for (const a of this.bipedAgents) a.brain.activationFn = patch.activationFn;
      for (const a of this.simpleAgents) a.brain.activationFn = patch.activationFn;
      if (this.bestGenomeEver) {
        this.bestGenomeEver = { ...this.bestGenomeEver, activationFn: patch.activationFn };
      }
    }
    this.emitTelemetry();
  }

  public applyPersistedModeState(state: PersistedModeState): void {
    this.mode = state.mode;
    this.generation = state.generation;
    this.bestDistanceEver = state.bestDistanceEver;
    this.bestFitnessEver = state.bestFitnessEver;
    this.milestone30mReached = state.milestone30mReached;
    this.history = state.history ? [...state.history] : [];
    this.bestGenomeEver = state.bestGenome;
    this.hyperparams = {
      ...this.hyperparams,
      activationFn: state.activationFn || 'tanh',
      populationSize: state.populationSize || 12,
      mutationRate: state.mutationRate || 0.16,
      mutationStrength: state.mutationStrength || 0.24,
    };

    const seeds = this.buildSeedsFromGenome(state.bestGenome, this.hyperparams.populationSize);
    this.initMode(state.mode, seeds, true);
  }

  public loadCheckpointGenome(
    mode: SimulationMode,
    genome: GenomeData,
    generation: number,
    activationFn: EvolutionHyperparams['activationFn'],
  ): void {
    this.mode = mode;
    this.generation = Math.max(1, generation);
    this.bestGenomeEver = genome;
    this.hyperparams = { ...this.hyperparams, activationFn };
    const seeds = this.buildSeedsFromGenome(genome, this.hyperparams.populationSize);
    this.initMode(mode, seeds, true);
  }

  private buildSeedsFromGenome(genome: GenomeData, popSize: number): NeuralNetwork[] {
    const base = NeuralNetwork.deserialize(genome);
    base.activationFn = this.hyperparams.activationFn;
    const seeds: NeuralNetwork[] = [base];
    for (let i = 1; i < popSize; i++) {
      const v = base.clone();
      v.mutate(this.hyperparams.mutationRate, this.hyperparams.mutationStrength * 0.85);
      seeds.push(v);
    }
    return seeds;
  }

  public getExportableState(): PersistedGenerationPayload | null {
    const genome = this.getCurrentBestGenome();
    if (!genome) return null;
    return {
      mode: this.mode,
      generation: this.generation,
      bestDistanceEver: this.bestDistanceEver,
      bestFitnessEver: this.bestFitnessEver,
      milestone30mReached: this.milestone30mReached,
      hyperparams: { ...this.hyperparams },
      bestGenome: genome,
      history: [...this.history],
    };
  }

  private getCurrentBestGenome(): GenomeData | null {
    const isBiped = this.mode === 'biomechanics';
    const agents = isBiped ? this.bipedAgents : this.simpleAgents;
    if (agents.length > 0) {
      const sorted = [...agents].sort((a, b) => b.fitness - a.fitness);
      if (!this.bestGenomeEver || sorted[0].fitness >= this.bestFitnessEver * 0.9) {
        return sorted[0].brain.serialize();
      }
    }
    return this.bestGenomeEver;
  }

  public initMode(
    mode: SimulationMode,
    seedNetworks?: NeuralNetwork[],
    preserveStats = false,
  ): void {
    this.clearPopulation();
    this.clearEnvironmentMeshes();
    this.initPhysicsWorld();

    if (!seedNetworks && !preserveStats) {
      this.generation = 1;
      this.history = [];
      this.bestDistanceEver = 0;
      this.bestFitnessEver = 0;
      this.milestone30mReached = false;
      this.bestGenomeEver = null;
    }
    this.elapsedTime = 0;

    if (mode === 'biomechanics') {
      this.generationDuration = 28;
      this.buildBiomechanicsTrack();
      this.spawnBipedPopulation(seedNetworks);
      if (this.challenges.unevenTerrain) {
        this.rebuildUnevenTerrain(true);
      }
      if (!seedNetworks) {
        this.camera.position.set(4.5, 2.5, -2.5);
        this.controls.target.set(0, 1.0, 0.8);
      }
    } else if (mode === 'navigation') {
      this.generationDuration = 22;
      this.buildNavigationArena();
      this.spawnRoverPopulation(seedNetworks);
      if (!seedNetworks) {
        this.camera.position.set(0, 16, -15);
        this.controls.target.set(0, 0.3, 0);
      }
    } else if (mode === 'balance') {
      this.generationDuration = 18;
      this.buildBalanceArena();
      this.spawnBalancePopulation(seedNetworks);
      if (!seedNetworks) {
        this.camera.position.set(8.5, 3.2, 6.0);
        this.controls.target.set(0, 0.9, 0);
      }
    } else if (mode === 'parkour') {
      this.generationDuration = 22;
      this.buildParkourTrack();
      this.spawnParkourPopulation(seedNetworks);
      if (!seedNetworks) {
        this.camera.position.set(5.5, 3.0, -3.5);
        this.controls.target.set(0, 0.6, 2.0);
      }
    } else if (mode === 'manual_car') {
      this.generationDuration = 26;
      this.buildManualCarTrack();
      this.spawnManualCarPopulation(seedNetworks);
      if (!seedNetworks) {
        this.camera.position.set(-7.5, 5.2, -9.5);
        this.controls.target.set(0, 0.4, 3.0);
      }
    } else {
      // drone
      this.generationDuration = 22;
      this.buildDroneArena();
      this.spawnDronePopulation(seedNetworks);
      if (!seedNetworks) {
        this.camera.position.set(6.5, 5.2, -8.5);
        this.controls.target.set(0, 3.0, 4.0);
      }
    }

    this.windGroup.visible = mode === 'biomechanics' && this.challenges.headwind;
    this.emitTelemetry();
  }

  private buildBiomechanicsTrack(): void {
    const outerFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(240, 260),
      new THREE.MeshStandardMaterial({ color: 0x0a0e15, roughness: 0.95, metalness: 0.05 }),
    );
    outerFloor.rotation.x = -Math.PI / 2;
    outerFloor.position.set(0, -0.02, 55);
    outerFloor.receiveShadow = true;
    this.envGroup.add(outerFloor);

    const trackTex = this.assets.trackTexture.clone();
    trackTex.needsUpdate = true;
    trackTex.repeat.set(2, 70);
    const trackMat = new THREE.MeshStandardMaterial({
      map: trackTex,
      roughness: 0.82,
      metalness: 0.1,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const trackMesh = new THREE.Mesh(new THREE.PlaneGeometry(8, 140), trackMat);
    trackMesh.rotation.x = -Math.PI / 2;
    trackMesh.position.set(0, 0.0, 62);
    trackMesh.receiveShadow = true;
    this.envGroup.add(trackMesh);

    const curbMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.55,
      metalness: 0.4,
    });
    for (const sx of [-4.15, 4.15]) {
      const curb = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.1, 140), curbMat);
      curb.position.set(sx, 0.05, 62);
      curb.receiveShadow = true;
      this.envGroup.add(curb);
    }

    const milestones = [1, 5, 10, 20, 30, 50, 75, 100];
    for (const m of milestones) {
      const is30 = m === 30;
      const tex = createDistanceMarkerTexture(`${m} M`, is30);
      const plaqueMat = new THREE.MeshStandardMaterial({
        map: tex,
        roughness: 0.4,
        metalness: 0.2,
      });
      const plaque = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.42, 0.08), plaqueMat);
      plaque.position.set(-4.35, 0.95, m);
      plaque.rotation.y = Math.PI * 0.22;
      this.envGroup.add(plaque);

      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.0, 8), curbMat);
      post.position.set(-4.35, 0.5, m);
      this.envGroup.add(post);

      const lineMat = new THREE.MeshBasicMaterial({
        color: is30 ? 0xf59e0b : 0x38bdf8,
        transparent: true,
        opacity: is30 ? 0.75 : 0.35,
      });
      const line = new THREE.Mesh(new THREE.PlaneGeometry(7.8, is30 ? 0.16 : 0.07), lineMat);
      line.rotation.x = -Math.PI / 2;
      line.position.set(0, 0.008, m);
      this.envGroup.add(line);

      if (is30) {
        const archMat = new THREE.MeshStandardMaterial({
          color: 0xf59e0b,
          emissive: 0x92400e,
          emissiveIntensity: 0.6,
          roughness: 0.3,
          metalness: 0.7,
        });
        const leftPillar = new THREE.Mesh(new THREE.BoxGeometry(0.18, 3.4, 0.18), archMat);
        leftPillar.position.set(-4.1, 1.7, 30);
        const rightPillar = new THREE.Mesh(new THREE.BoxGeometry(0.18, 3.4, 0.18), archMat);
        rightPillar.position.set(4.1, 1.7, 30);
        const topBeam = new THREE.Mesh(new THREE.BoxGeometry(8.4, 0.22, 0.22), archMat);
        topBeam.position.set(0, 3.4, 30);
        this.envGroup.add(leftPillar, rightPillar, topBeam);
      }
    }
  }

  private buildNavigationArena(): void {
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x111722,
      roughness: 0.85,
      metalness: 0.1,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(24, 30), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.envGroup.add(floor);

    this.roverObstacles = [
      { x: -2.2, z: -2.0, r: 0.75 },
      { x: 2.0, z: -1.5, r: 0.75 },
      { x: 0.0, z: 1.5, r: 0.9 },
      { x: -3.5, z: 3.5, r: 0.8 },
      { x: 3.4, z: 4.0, r: 0.8 },
      { x: -1.2, z: 6.5, r: 0.75 },
      { x: 1.6, z: 7.2, r: 0.75 },
    ];
    this.roverBeacons = [
      new THREE.Vector3(0, 0.3, 4.5),
      new THREE.Vector3(-4.5, 0.3, 8.5),
      new THREE.Vector3(4.5, 0.3, 9.5),
      new THREE.Vector3(0, 0.3, -4.0),
    ];

    const obsMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.45,
      metalness: 0.6,
    });
    for (const obs of this.roverObstacles) {
      const b = new CANNON.Body({
        mass: 0,
        position: new CANNON.Vec3(obs.x, 0.75, obs.z),
        collisionFilterGroup: COLLISION_GROUP_WORLD,
        collisionFilterMask: COLLISION_GROUP_ROBOT,
      });
      b.addShape(new CANNON.Cylinder(obs.r, obs.r, 1.5, 14));
      this.world.addBody(b);

      const m = new THREE.Mesh(new THREE.CylinderGeometry(obs.r, obs.r, 1.5, 18), obsMat);
      m.position.set(obs.x, 0.75, obs.z);
      m.castShadow = true;
      m.receiveShadow = true;
      this.envGroup.add(m);
    }

    const beaconMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xd97706,
      emissiveIntensity: 1.6,
      roughness: 0.25,
    });
    for (const bp of this.roverBeacons) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.05, 10, 24), beaconMat);
      ring.rotation.x = Math.PI / 2;
      ring.position.set(bp.x, 0.04, bp.z);
      this.envGroup.add(ring);
    }
  }

  private buildBalanceArena(): void {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(52, 32),
      new THREE.MeshStandardMaterial({ color: 0x0d121b, roughness: 0.9, metalness: 0.05 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.envGroup.add(floor);

    const railMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      roughness: 0.3,
      metalness: 0.85,
    });
    const count = Math.min(16, Math.max(6, this.hyperparams.populationSize));
    const half = (count - 1) * 0.5;
    for (let i = 0; i < count; i++) {
      const lx = (i - half) * 1.85;
      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.05, 14.0), railMat);
      rail.position.set(lx, 0.025, 0);
      this.envGroup.add(rail);
    }
  }

  private buildParkourTrack(): void {
    const trackTex = this.assets.trackTexture.clone();
    trackTex.needsUpdate = true;
    trackTex.repeat.set(2, 50);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(7.5, 100),
      new THREE.MeshStandardMaterial({ map: trackTex, roughness: 0.85, metalness: 0.1 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(0, 0, 44);
    floor.receiveShadow = true;
    this.envGroup.add(floor);

    this.parkourHurdles = [
      { z: 7, h: 0.45 },
      { z: 14, h: 0.55 },
      { z: 22, h: 0.5 },
      { z: 30, h: 0.65 },
      { z: 39, h: 0.55 },
      { z: 48, h: 0.7 },
      { z: 58, h: 0.6 },
      { z: 68, h: 0.7 },
    ];
    const hMat = new THREE.MeshStandardMaterial({
      map: this.assets.hazardTexture,
      roughness: 0.6,
      metalness: 0.25,
    });
    for (const h of this.parkourHurdles) {
      const b = new CANNON.Body({
        mass: 0,
        position: new CANNON.Vec3(0, h.h / 2, h.z),
        collisionFilterGroup: COLLISION_GROUP_WORLD,
        collisionFilterMask: COLLISION_GROUP_ROBOT,
      });
      b.addShape(new CANNON.Box(new CANNON.Vec3(3.6, h.h / 2, 0.22)));
      this.world.addBody(b);

      const m = new THREE.Mesh(new THREE.BoxGeometry(7.2, h.h, 0.44), hMat);
      m.position.set(0, h.h / 2, h.z);
      m.castShadow = true;
      m.receiveShadow = true;
      this.envGroup.add(m);
    }
  }

  private buildManualCarTrack(): void {
    this.carCircuit = createManualCarCircuitData();
    const { samples, halfWidth } = this.carCircuit;

    const outerFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(240, 240),
      new THREE.MeshStandardMaterial({ color: 0x090d14, roughness: 0.95, metalness: 0.05 }),
    );
    outerFloor.rotation.x = -Math.PI / 2;
    outerFloor.position.set(24, -0.02, 12);
    outerFloor.receiveShadow = true;
    this.envGroup.add(outerFloor);

    // Build continuous closed asphalt track ribbon geometry
    const n = samples.length;
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    for (let i = 0; i <= n; i++) {
      const sp = samples[i % n];
      const lx = sp.x - sp.nx * halfWidth;
      const lz = sp.z - sp.nz * halfWidth;
      const rx = sp.x + sp.nx * halfWidth;
      const rz = sp.z + sp.nz * halfWidth;
      positions.push(lx, 0.01, lz, rx, 0.01, rz);
      const v = i * 0.45;
      uvs.push(0, v, 1, v);

      if (i < n) {
        const base = i * 2;
        indices.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
      }
    }

    const roadGeo = new THREE.BufferGeometry();
    roadGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    roadGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    roadGeo.setIndex(indices);
    roadGeo.computeVertexNormals();

    const roadTex = this.assets.trackTexture.clone();
    roadTex.needsUpdate = true;
    const roadMat = new THREE.MeshStandardMaterial({
      map: roadTex,
      roughness: 0.8,
      metalness: 0.08,
      side: THREE.DoubleSide,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    roadMesh.receiveShadow = true;
    this.envGroup.add(roadMesh);

    // Curbs & Apex Markers around the circuit
    const redMat = new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.6 });
    const whiteMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.6 });
    for (let i = 0; i < n; i += 2) {
      const sp = samples[i];
      const curbMat = (i / 2) % 2 === 0 ? redMat : whiteMat;
      const ang = Math.atan2(sp.tx, sp.tz);
      for (const side of [-1, 1]) {
        const curb = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.08, 1.35), curbMat);
        curb.position.set(
          sp.x + sp.nx * side * (halfWidth + 0.18),
          0.04,
          sp.z + sp.nz * side * (halfWidth + 0.18),
        );
        curb.rotation.y = ang;
        this.envGroup.add(curb);
      }
    }

    // Start/Finish Gantry Arch
    const start = samples[0];
    const gantryMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0x92400e,
      emissiveIntensity: 0.65,
      metalness: 0.6,
      roughness: 0.3,
    });
    const startAng = Math.atan2(start.tx, start.tz);
    const gantry = new THREE.Group();
    gantry.position.set(start.x, 0, start.z);
    gantry.rotation.y = startAng;
    const lp = new THREE.Mesh(new THREE.BoxGeometry(0.24, 3.8, 0.24), gantryMat);
    lp.position.set(-halfWidth - 0.35, 1.9, 0);
    const rp = new THREE.Mesh(new THREE.BoxGeometry(0.24, 3.8, 0.24), gantryMat);
    rp.position.set(halfWidth + 0.35, 1.9, 0);
    const beam = new THREE.Mesh(new THREE.BoxGeometry(halfWidth * 2 + 1.0, 0.3, 0.3), gantryMat);
    beam.position.set(0, 3.8, 0);
    gantry.add(lp, rp, beam);
    this.envGroup.add(gantry);
  }

  private buildDroneArena(): void {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(160, 180),
      new THREE.MeshStandardMaterial({ color: 0x0a0f18, roughness: 0.9, metalness: 0.1 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(0, 0, 32);
    floor.receiveShadow = true;
    this.envGroup.add(floor);

    const grid = new THREE.GridHelper(120, 60, 0x38bdf8, 0x1e293b);
    grid.position.set(0, 0.02, 32);
    this.envGroup.add(grid);

    this.droneCheckpoints = [
      new THREE.Vector3(0, 3.0, 5),
      new THREE.Vector3(2.2, 3.8, 12),
      new THREE.Vector3(-2.0, 4.4, 20),
      new THREE.Vector3(1.5, 3.6, 28),
      new THREE.Vector3(0, 4.8, 36),
      new THREE.Vector3(-1.8, 4.0, 44),
      new THREE.Vector3(2.0, 4.5, 54),
      new THREE.Vector3(0, 3.5, 64),
    ];

    const ringMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 1.4,
      roughness: 0.25,
    });
    const pylonMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.5,
      metalness: 0.6,
    });
    for (const cp of this.droneCheckpoints) {
      const torus = new THREE.Mesh(new THREE.TorusGeometry(1.15, 0.08, 12, 32), ringMat);
      torus.position.copy(cp);
      this.envGroup.add(torus);

      const postH = Math.max(0.2, cp.y - 1.15);
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, postH, 8), pylonMat);
      post.position.set(cp.x, postH * 0.5, cp.z);
      this.envGroup.add(post);
    }
  }

  private spawnBipedPopulation(seedNetworks?: NeuralNetwork[]): void {
    const popSize = this.hyperparams.populationSize;
    const mats = { chassisMaterial: this.chassisMat, footMaterial: this.footMat };

    for (let i = 0; i < popSize; i++) {
      let brain: NeuralNetwork;
      if (seedNetworks && seedNetworks[i]) {
        brain = seedNetworks[i];
      } else {
        brain = new NeuralNetwork(18, 16, 6, this.hyperparams.activationFn);
        for (let h = 0; h < 6; h++) {
          brain.wIH[h * 18 + 15] += (Math.random() - 0.5) * 0.9;
          brain.wIH[h * 18 + 16] += (Math.random() - 0.5) * 0.9;
        }
      }

      const walker = new BipedWalker3D(
        i,
        brain,
        this.world,
        this.groundBody,
        mats,
        this.assets,
        new CANNON.Vec3(0, 0, 0),
        false,
      );
      this.scene.add(walker.rootGroup);
      this.bipedAgents.push(walker);
    }
    this.trackedId = 0;
  }

  private spawnRoverPopulation(seedNetworks?: NeuralNetwork[]): void {
    const popSize = this.hyperparams.populationSize;
    for (let i = 0; i < popSize; i++) {
      const brain = seedNetworks?.[i] ?? new NeuralNetwork(9, 10, 2, this.hyperparams.activationFn);
      const agent = new RoverAgent3D(
        i,
        brain,
        this.world,
        this.chassisMat,
        this.assets,
        this.roverObstacles,
        this.roverBeacons,
      );
      this.scene.add(agent.rootGroup);
      this.simpleAgents.push(agent);
    }
    this.trackedId = 0;
  }

  private spawnBalancePopulation(seedNetworks?: NeuralNetwork[]): void {
    const popSize = Math.min(16, Math.max(6, this.hyperparams.populationSize));
    const half = (popSize - 1) * 0.5;
    for (let i = 0; i < popSize; i++) {
      let brain: NeuralNetwork;
      if (seedNetworks?.[i]) {
        brain = seedNetworks[i];
      } else {
        brain = new NeuralNetwork(6, 8, 1, this.hyperparams.activationFn);
        // Smart Gen 1 prior: wire tilt & angular velocity slightly positively for 3 agents so carts move immediately
        if (i < 3) {
          brain.wIH[0 * 6 + 2] = 1.4 + (Math.random() - 0.5) * 0.8;
          brain.wIH[0 * 6 + 3] = 0.6 + (Math.random() - 0.5) * 0.5;
          brain.wHO[0] = 1.1 + (Math.random() - 0.5) * 0.6;
        }
      }
      const laneX = (i - half) * 1.85;
      const agent = new CartPoleAgent3D(i, brain, this.world, this.chassisMat, this.assets, laneX);
      this.scene.add(agent.rootGroup);
      this.simpleAgents.push(agent);
    }
    this.trackedId = 0;
  }

  private spawnParkourPopulation(seedNetworks?: NeuralNetwork[]): void {
    const popSize = this.hyperparams.populationSize;
    for (let i = 0; i < popSize; i++) {
      const brain = seedNetworks?.[i] ?? new NeuralNetwork(8, 10, 3, this.hyperparams.activationFn);
      const agent = new ParkourAgent3D(
        i,
        brain,
        this.world,
        this.chassisMat,
        this.assets,
        this.parkourHurdles,
      );
      this.scene.add(agent.rootGroup);
      this.simpleAgents.push(agent);
    }
    this.trackedId = 0;
  }

  private spawnManualCarPopulation(seedNetworks?: NeuralNetwork[]): void {
    const popSize = this.hyperparams.populationSize;
    for (let i = 0; i < popSize; i++) {
      let brain: NeuralNetwork;
      if (seedNetworks?.[i]) {
        brain = seedNetworks[i];
      } else {
        brain = new NeuralNetwork(16, 18, 5, this.hyperparams.activationFn);
        // Mild Gen 1 prior so initial cars shift out of Neutral & roll forward while exploring steering/clutch
        if (i < 3) {
          brain.bO[1] += 0.5; // gentle throttle bias
          brain.bO[3] -= 0.6; // clutch engagement bias
          brain.bO[4] += 0.4; // gear upshift bias
          brain.wIH[0 * 16 + 5] += 0.65;
          brain.wIH[0 * 16 + 6] -= 0.75;
          brain.wHO[0] += 0.85;
        }
      }
      const agent = new ManualCarAgent3D(
        i,
        brain,
        this.world,
        this.chassisMat,
        this.assets,
        this.carCircuit,
        false,
      );
      this.scene.add(agent.rootGroup);
      this.simpleAgents.push(agent);
    }
    this.trackedId = 0;
  }

  private spawnDronePopulation(seedNetworks?: NeuralNetwork[]): void {
    const popSize = this.hyperparams.populationSize;
    for (let i = 0; i < popSize; i++) {
      let brain: NeuralNetwork;
      if (seedNetworks?.[i]) {
        brain = seedNetworks[i];
      } else {
        brain = new NeuralNetwork(12, 12, 4, this.hyperparams.activationFn);
        if (i < 3) {
          // Gentle collective lift bias in Gen 1 so quadcopters explore airborne control immediately
          brain.wIH[0 * 12 + 1] += 0.6;
          brain.wIH[0 * 12 + 4] -= 0.45;
          for (let o = 0; o < 4; o++) brain.wHO[o * 12 + 0] += 0.4;
        }
      }
      const agent = new DroneAgent3D(
        i,
        brain,
        this.world,
        this.chassisMat,
        this.assets,
        this.droneCheckpoints,
        false,
      );
      this.scene.add(agent.rootGroup);
      this.simpleAgents.push(agent);
    }
    this.trackedId = 0;
  }

  public loadPretrainedBipedGait(): void {
    this.mode = 'biomechanics';
    const base = createPretrainedBipedNetwork();
    this.hyperparams = { ...this.hyperparams, activationFn: base.activationFn };
    this.bestGenomeEver = base.serialize();
    const seeds = this.buildSeedsFromGenome(this.bestGenomeEver, this.hyperparams.populationSize);
    this.initMode('biomechanics', seeds, true);
  }

  public loadPretrainedCartPole(): void {
    this.mode = 'balance';
    const base = createPretrainedCartPoleNetwork(this.hyperparams.activationFn);
    this.bestGenomeEver = base.serialize();
    const popSize = Math.min(16, Math.max(6, this.hyperparams.populationSize));
    const seeds = this.buildSeedsFromGenome(this.bestGenomeEver, popSize);
    this.initMode('balance', seeds, true);
  }

  public loadPretrainedManualCar(): void {
    this.mode = 'manual_car';
    const base = createPretrainedManualCarNetwork(this.hyperparams.activationFn);
    this.bestGenomeEver = base.serialize();
    const seeds = this.buildSeedsFromGenome(this.bestGenomeEver, this.hyperparams.populationSize);
    this.initMode('manual_car', seeds, true);
  }

  public loadPretrainedDrone(): void {
    this.mode = 'drone';
    const base = createPretrainedDroneNetwork(this.hyperparams.activationFn);
    this.bestGenomeEver = base.serialize();
    const seeds = this.buildSeedsFromGenome(this.bestGenomeEver, this.hyperparams.populationSize);
    this.initMode('drone', seeds, true);
  }

  public setChallenges(next: Partial<EnvironmentChallenges>): void {
    const prevTerrain = this.challenges.unevenTerrain;
    this.challenges = { ...this.challenges, ...next };
    if (this.mode === 'biomechanics' && prevTerrain !== this.challenges.unevenTerrain) {
      this.rebuildUnevenTerrain(this.challenges.unevenTerrain);
    }
    this.windGroup.visible = this.mode === 'biomechanics' && this.challenges.headwind;
  }

  private rebuildUnevenTerrain(enabled: boolean): void {
    for (const obs of this.terrainObstacles) {
      this.world.removeBody(obs.body);
      this.challengeGroup.remove(obs.mesh);
      obs.mesh.geometry.dispose();
      (obs.mesh.material as THREE.Material).dispose();
    }
    this.terrainObstacles = [];

    if (!enabled) return;

    const mat = new THREE.MeshStandardMaterial({
      map: this.assets.hazardTexture,
      roughness: 0.7,
      metalness: 0.25,
    });

    for (let z = 4.5; z <= 96; z += 4.2) {
      const isStep = Math.round(z) % 2 === 0;
      const height = isStep ? 0.045 + ((z * 13) % 3) * 0.015 : 0.038;
      const depth = isStep ? 0.36 : 0.55;
      const width = 6.2;

      const body = new CANNON.Body({
        mass: 0,
        material: this.groundMat,
        position: new CANNON.Vec3(0, height * 0.5, z),
        collisionFilterGroup: COLLISION_GROUP_WORLD,
        collisionFilterMask: COLLISION_GROUP_ROBOT | COLLISION_GROUP_PROJECTILE,
      });
      body.addShape(new CANNON.Box(new CANNON.Vec3(width * 0.5, height * 0.5, depth * 0.5)));
      this.world.addBody(body);

      const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), mat.clone());
      mesh.position.set(0, height * 0.5, z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.challengeGroup.add(mesh);

      this.terrainObstacles.push({ body, mesh });
    }
    mat.dispose();
  }

  public throwBallAtLeader(): void {
    if (this.mode !== 'biomechanics') return;
    const leader = this.getTrackedBiped();
    if (!leader || !leader.alive) return;

    const targetPos = leader.torso.position;
    const side = Math.random() < 0.5 ? -1 : 1;
    const spawnX = targetPos.x + side * 2.8;
    const spawnY = targetPos.y + 0.55;
    const spawnZ = targetPos.z + 3.2;

    const radius = 0.22;
    const body = new CANNON.Body({
      mass: 4.5,
      position: new CANNON.Vec3(spawnX, spawnY, spawnZ),
      material: this.chassisMat,
      collisionFilterGroup: COLLISION_GROUP_PROJECTILE,
      collisionFilterMask: COLLISION_GROUP_WORLD | COLLISION_GROUP_ROBOT,
    });
    body.addShape(new CANNON.Sphere(radius));

    const timeToHit = 0.38;
    const vx = (targetPos.x - spawnX) / timeToHit;
    const vz = (targetPos.z + leader.torso.velocity.z * timeToHit - spawnZ) / timeToHit;
    const vy = (targetPos.y - spawnY + 0.5 * 9.81 * timeToHit * timeToHit) / timeToHit;
    body.velocity.set(vx, vy, vz);
    this.world.addBody(body);

    const ballMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0x7f1d1d,
      emissiveIntensity: 0.4,
      roughness: 0.3,
      metalness: 0.4,
    });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 14), ballMat);
    mesh.castShadow = true;
    mesh.position.set(spawnX, spawnY, spawnZ);
    this.scene.add(mesh);

    this.projectiles.push({ body, mesh, age: 0 });
    this.challenges.ballsThrown += 1;
    this.emitTelemetry();
  }

  private getTrackedBiped(): BipedWalker3D | null {
    let bestAlive: BipedWalker3D | null = null;
    for (const a of this.bipedAgents) {
      if (a.alive && (!bestAlive || a.distance > bestAlive.distance)) {
        bestAlive = a;
      }
    }
    if (bestAlive) return bestAlive;

    let bestDead: BipedWalker3D | null = null;
    for (const a of this.bipedAgents) {
      if (!bestDead || a.fitness > bestDead.fitness) {
        bestDead = a;
      }
    }
    return bestDead;
  }

  private getTrackedSimple(): SimpleModeAgent | null {
    let bestAlive: SimpleModeAgent | null = null;
    for (const a of this.simpleAgents) {
      if (a.alive && (!bestAlive || a.fitness > bestAlive.fitness)) {
        bestAlive = a;
      }
    }
    if (bestAlive) return bestAlive;
    return this.simpleAgents[0] ?? null;
  }

  public setPaused(p: boolean): void {
    this.paused = p;
  }

  public setSimSpeed(speed: number): void {
    this.simSpeed = speed;
  }

  public setFollowLeader(follow: boolean): void {
    this.followLeader = follow;
  }

  public nextGenerationNow(): void {
    this.evolveNextGeneration();
  }

  public resetEvolution(): void {
    this.initMode(this.mode, undefined, false);
  }

  private evolveNextGeneration(): void {
    const isBiped = this.mode === 'biomechanics';
    const agents = isBiped ? this.bipedAgents : this.simpleAgents;
    if (agents.length === 0) return;

    const sorted = [...agents].sort((a, b) => b.fitness - a.fitness);
    const bestFit = sorted[0].fitness;
    const avgFit = sorted.reduce((acc, a) => acc + a.fitness, 0) / sorted.length;
    const bestDist = Math.max(...sorted.map((a) => a.maxDistance));

    if (bestDist > this.bestDistanceEver) this.bestDistanceEver = bestDist;
    if (bestFit >= this.bestFitnessEver || !this.bestGenomeEver) {
      if (bestFit > this.bestFitnessEver) this.bestFitnessEver = bestFit;
      this.bestGenomeEver = sorted[0].brain.serialize();
    }
    if (isBiped && this.bestDistanceEver >= 30) {
      this.milestone30mReached = true;
    }

    this.history.push({
      generation: this.generation,
      bestFitness: Number(bestFit.toFixed(1)),
      avgFitness: Number(avgFit.toFixed(1)),
      bestDistance: Number(bestDist.toFixed(2)),
    });
    if (this.history.length > 40) this.history.shift();

    const targetPop =
      this.mode === 'balance'
        ? Math.min(16, Math.max(6, this.hyperparams.populationSize))
        : this.hyperparams.populationSize;

    const nextBrains: NeuralNetwork[] = [];
    // Hall-of-Fame elitism preserves all-time best genome across stochastic drops
    if (
      this.hyperparams.hallOfFame &&
      this.bestGenomeEver &&
      this.bestGenomeEver.inputSize === sorted[0].brain.inputSize &&
      this.bestGenomeEver.hiddenSize === sorted[0].brain.hiddenSize &&
      this.bestGenomeEver.outputSize === sorted[0].brain.outputSize
    ) {
      const hofNet = NeuralNetwork.deserialize(this.bestGenomeEver);
      hofNet.activationFn = this.hyperparams.activationFn;
      nextBrains.push(hofNet);
    }

    const eliteCount = Math.max(1, this.hyperparams.elitismCount);
    for (let i = 0; i < eliteCount && i < sorted.length && nextBrains.length < targetPop; i++) {
      const elite = sorted[i].brain.clone();
      elite.activationFn = this.hyperparams.activationFn;
      nextBrains.push(elite);
    }

    const pickParent = () => {
      const poolSize = Math.max(2, Math.ceil(sorted.length * 0.45));
      const a = sorted[Math.floor(Math.random() * poolSize)];
      const b = sorted[Math.floor(Math.random() * poolSize)];
      return a.fitness >= b.fitness ? a.brain : b.brain;
    };

    const mutRate = this.hyperparams.mutationRate;
    const mutStrength = this.hyperparams.mutationStrength;

    while (nextBrains.length < targetPop) {
      const child =
        Math.random() < 0.7
          ? NeuralNetwork.crossover(pickParent(), pickParent())
          : pickParent().clone();
      child.activationFn = this.hyperparams.activationFn;
      child.mutate(mutRate, mutStrength);
      nextBrains.push(child);
    }

    const completedPayload: PersistedGenerationPayload = {
      mode: this.mode,
      generation: this.generation + 1,
      bestDistanceEver: this.bestDistanceEver,
      bestFitnessEver: this.bestFitnessEver,
      milestone30mReached: this.milestone30mReached,
      hyperparams: { ...this.hyperparams },
      bestGenome: this.bestGenomeEver ?? sorted[0].brain.serialize(),
      history: [...this.history],
    };

    this.generation += 1;
    this.initMode(this.mode, nextBrains, true);
    this.onGenerationComplete?.(completedPayload);
  }

  private stepSimulation(dt: number): void {
    const fixedDt = 1 / 60;
    const headwind = this.mode === 'biomechanics' && this.challenges.headwind ? this.challenges.windForce : 0;

    this.world.step(fixedDt);

    if (this.mode === 'biomechanics') {
      let aliveCount = 0;
      for (const a of this.bipedAgents) {
        if (a.alive) {
          a.updateFootContacts(this.world);
          a.stepControl(fixedDt, headwind);
          if (a.alive) aliveCount++;
        }
        a.syncVisuals();
      }

      const leader = this.getTrackedBiped();
      if (leader) {
        this.trackedId = leader.id;
        for (const a of this.bipedAgents) {
          a.setHighlighted(a.id === leader.id);
        }
        if (leader.maxDistance > this.bestDistanceEver) {
          this.bestDistanceEver = leader.maxDistance;
          if (this.bestDistanceEver >= 30) this.milestone30mReached = true;
        }
        if (leader.fitness > this.bestFitnessEver) {
          this.bestFitnessEver = leader.fitness;
          this.bestGenomeEver = leader.brain.serialize();
        }
      }

      for (let i = this.projectiles.length - 1; i >= 0; i--) {
        const p = this.projectiles[i];
        p.age += fixedDt;
        p.mesh.position.set(p.body.position.x, p.body.position.y, p.body.position.z);
        p.mesh.quaternion.set(p.body.quaternion.x, p.body.quaternion.y, p.body.quaternion.z, p.body.quaternion.w);
        if (p.age > 6) {
          this.world.removeBody(p.body);
          this.scene.remove(p.mesh);
          p.mesh.geometry.dispose();
          (p.mesh.material as THREE.Material).dispose();
          this.projectiles.splice(i, 1);
        }
      }

      this.elapsedTime += fixedDt;
      if (aliveCount === 0 || this.elapsedTime >= this.generationDuration) {
        this.evolveNextGeneration();
      }
    } else {
      let aliveCount = 0;
      for (const a of this.simpleAgents) {
        if (a.alive) {
          a.stepControl(fixedDt);
          if (a.alive) aliveCount++;
        }
        a.syncVisuals();
      }
      const leader = this.getTrackedSimple();
      if (leader) {
        this.trackedId = leader.id;
        for (const a of this.simpleAgents) {
          a.setHighlighted(a.id === leader.id);
        }
        if (leader.maxDistance > this.bestDistanceEver) {
          this.bestDistanceEver = leader.maxDistance;
        }
        if (leader.fitness > this.bestFitnessEver) {
          this.bestFitnessEver = leader.fitness;
          this.bestGenomeEver = leader.brain.serialize();
        }
      }

      this.elapsedTime += fixedDt;
      if (aliveCount === 0 || this.elapsedTime >= this.generationDuration) {
        this.evolveNextGeneration();
      }
    }
    void dt;
  }

  private updateCameraAndWind(dt: number): void {
    let targetX = 0;
    let targetY = 1.0;
    let targetZ = 0;

    if (this.mode === 'biomechanics') {
      const leader = this.getTrackedBiped();
      if (leader) {
        targetX = leader.torso.position.x * 0.4;
        targetY = Math.max(0.85, leader.torso.position.y * 0.88);
        targetZ = leader.torso.position.z + 0.4;
      }
    } else {
      const leader = this.getTrackedSimple();
      if (leader) {
        const p = leader.getTrackPosition();
        targetX = p.x;
        targetY = Math.max(0.5, p.y);
        targetZ = p.z;
      }
    }

    if (this.followLeader && !this.userOrbiting) {
      const dampLambda = 6.0;
      const nx = THREE.MathUtils.damp(this.controls.target.x, targetX, dampLambda, dt);
      const ny = THREE.MathUtils.damp(this.controls.target.y, targetY, dampLambda, dt);
      const nz = THREE.MathUtils.damp(this.controls.target.z, targetZ, dampLambda, dt);

      _camDelta.set(nx - this.controls.target.x, ny - this.controls.target.y, nz - this.controls.target.z);
      this.controls.target.set(nx, ny, nz);
      this.camera.position.add(_camDelta);
    }

    this.keyLight.position.set(targetX + 12, 18, targetZ + 8);
    this.keyLight.target.position.set(targetX, 0.5, targetZ);

    if (this.windGroup.visible) {
      const dummy = new THREE.Object3D();
      for (let i = 0; i < this.windPositions.length; i++) {
        const wp = this.windPositions[i];
        wp.z -= wp.speed * dt;
        if (wp.z < targetZ - 8) {
          wp.z = targetZ + 9 + Math.random() * 4;
          wp.x = (Math.random() - 0.5) * 6.5;
          wp.y = 0.4 + Math.random() * 2.2;
        }
        dummy.position.set(wp.x, wp.y, wp.z);
        dummy.updateMatrix();
        this.windStreaks.setMatrixAt(i, dummy.matrix);
      }
      this.windStreaks.instanceMatrix.needsUpdate = true;
    }

    this.controls.update();
  }

  private getModeLabels(mode: SimulationMode): { inputLabels: string[]; outputLabels: string[] } {
    switch (mode) {
      case 'navigation':
        return { inputLabels: ROVER_INPUT_LABELS, outputLabels: ROVER_OUTPUT_LABELS };
      case 'balance':
        return { inputLabels: BALANCE_INPUT_LABELS, outputLabels: BALANCE_OUTPUT_LABELS };
      case 'parkour':
        return { inputLabels: PARKOUR_INPUT_LABELS, outputLabels: PARKOUR_OUTPUT_LABELS };
      case 'manual_car':
        return { inputLabels: MANUAL_CAR_INPUT_LABELS, outputLabels: MANUAL_CAR_OUTPUT_LABELS };
      case 'drone':
        return { inputLabels: DRONE_INPUT_LABELS, outputLabels: DRONE_OUTPUT_LABELS };
      default:
        return { inputLabels: BIPED_INPUT_LABELS, outputLabels: BIPED_OUTPUT_LABELS };
    }
  }

  private emitTelemetry(): void {
    const isBiped = this.mode === 'biomechanics';
    const snapshots: AgentSnapshot[] = isBiped
      ? this.bipedAgents.map((a) => a.getSnapshot())
      : this.simpleAgents.map((a) => a.getSnapshot());

    const aliveCount = snapshots.filter((s) => s.alive).length;
    const bestDistCur = snapshots.length ? Math.max(...snapshots.map((s) => s.maxDistance)) : 0;
    const bestFitCur = snapshots.length ? Math.max(...snapshots.map((s) => s.fitness)) : 0;
    const tracked = snapshots.find((s) => s.id === this.trackedId) ?? snapshots[0] ?? null;

    let networkData: NetworkInspectorData | null = null;
    if (isBiped) {
      const leader = this.getTrackedBiped();
      if (leader) {
        networkData = {
          inputs: Array.from(leader.brain.lastInputs),
          hidden: Array.from(leader.brain.lastHidden),
          outputs: Array.from(leader.brain.lastOutputs),
          inputLabels: BIPED_INPUT_LABELS,
          outputLabels: BIPED_OUTPUT_LABELS,
          weightsIH: Array.from(leader.brain.wIH),
          weightsHO: Array.from(leader.brain.wHO),
          activationFn: leader.brain.activationFn,
        };
      }
    } else {
      const leader = this.getTrackedSimple();
      if (leader) {
        const { inputLabels, outputLabels } = this.getModeLabels(this.mode);
        networkData = {
          inputs: Array.from(leader.brain.lastInputs),
          hidden: Array.from(leader.brain.lastHidden),
          outputs: Array.from(leader.brain.lastOutputs),
          inputLabels,
          outputLabels,
          weightsIH: Array.from(leader.brain.wIH),
          weightsHO: Array.from(leader.brain.wHO),
          activationFn: leader.brain.activationFn,
        };
      }
    }

    this.onTelemetry({
      mode: this.mode,
      generation: this.generation,
      aliveCount,
      totalCount: snapshots.length,
      elapsedTime: this.elapsedTime,
      generationDuration: this.generationDuration,
      bestDistanceCurrent: bestDistCur,
      bestDistanceEver: this.bestDistanceEver,
      bestFitnessCurrent: bestFitCur,
      bestFitnessEver: this.bestFitnessEver,
      milestone30mReached: this.milestone30mReached,
      trackedAgent: tracked,
      agents: snapshots,
      history: [...this.history],
      networkData,
      bestGenome: this.getCurrentBestGenome(),
      hyperparams: { ...this.hyperparams },
    });
  }

  private afterDraw(cb: () => void): void {
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    const poll = () => {
      if (this.disposed) return;
      const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
      if (state === gl.UNSIGNALED) {
        this.frameId = window.setTimeout(poll, 5);
      } else {
        if (sync) gl.deleteSync(sync);
        this.frameId = window.setTimeout(cb, 80);
      }
    };
    this.frameId = window.setTimeout(poll, 5);
  }

  private startLoop(): void {
    const tick = () => {
      if (this.disposed) return;
      const now = performance.now();
      const rawDt = Math.min((now - this.lastPerfTime) / 1000, 0.05);
      this.lastPerfTime = now;

      if (!this.paused) {
        const steps = this.isSoftwareGl ? Math.min(this.simSpeed, 3) : this.simSpeed;
        for (let s = 0; s < steps; s++) {
          this.stepSimulation(rawDt);
        }
      }

      this.updateCameraAndWind(rawDt);
      this.renderer.render(this.scene, this.camera);

      this.telemetryTimer += rawDt;
      if (this.telemetryTimer >= 0.08) {
        this.telemetryTimer = 0;
        this.emitTelemetry();
      }

      if (this.isSoftwareGl) {
        this.afterDraw(tick);
      } else {
        this.frameId = requestAnimationFrame(tick);
      }
    };

    if (this.isSoftwareGl) {
      this.afterDraw(tick);
    } else {
      this.frameId = requestAnimationFrame(tick);
    }
  }

  private handleResize = (): void => {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(this.isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
  };

  public dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.frameId);
    clearTimeout(this.frameId);
    window.removeEventListener('resize', this.handleResize);
    this.renderer.domElement.removeEventListener('webglcontextlost', this.handleContextLost);
    this.renderer.domElement.removeEventListener('webglcontextrestored', this.handleContextRestored);

    this.clearPopulation();
    this.clearEnvironmentMeshes();
    this.assets.dispose();
    this.controls.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}
