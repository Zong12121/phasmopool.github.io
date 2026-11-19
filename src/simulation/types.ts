export type SimulationMode =
  | 'navigation'
  | 'balance'
  | 'parkour'
  | 'biomechanics'
  | 'manual_car'
  | 'drone';

export type ActivationFn = 'tanh' | 'swish' | 'leaky_relu';

export interface GenomeData {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  activationFn: ActivationFn;
  wIH: number[];
  bH: number[];
  wHO: number[];
  bO: number[];
}

export interface EvolutionHyperparams {
  populationSize: number;
  mutationRate: number;
  mutationStrength: number;
  activationFn: ActivationFn;
  elitismCount: number;
  hallOfFame: boolean;
}

export interface EnvironmentChallenges {
  unevenTerrain: boolean;
  headwind: boolean;
  windForce: number; // Newtons applied horizontally (-Z)
  ballsThrown: number;
  manualUnlock30m: boolean;
}

export interface AgentSnapshot {
  id: number;
  alive: boolean;
  distance: number;
  maxDistance: number;
  fitness: number;
  speed: number;
  torsoHeight: number;
  pitchDeg: number;
  rollDeg: number;
  leftContact: boolean;
  rightContact: boolean;
  disqualifiedReason?: string;
  // Optional Mode 5 (Manual Car) telemetry
  rpm?: number;
  gear?: number; // -1 = R, 0 = N, 1..5
  clutch?: number; // 0..1 pedal pressed
  throttle?: number; // 0..1
  brake?: number; // 0..1
  stalled?: boolean;
  // Optional Mode 6 (Drone) telemetry
  checkpointsPassed?: number;
}

export interface GenerationHistoryPoint {
  generation: number;
  bestFitness: number;
  avgFitness: number;
  bestDistance: number;
}

export interface NetworkInspectorData {
  inputs: number[];
  hidden: number[];
  outputs: number[];
  inputLabels: string[];
  outputLabels: string[];
  weightsIH: number[];
  weightsHO: number[];
  activationFn?: ActivationFn;
}

export interface SimulationTelemetry {
  mode: SimulationMode;
  generation: number;
  aliveCount: number;
  totalCount: number;
  elapsedTime: number;
  generationDuration: number;
  bestDistanceCurrent: number;
  bestDistanceEver: number;
  bestFitnessCurrent: number;
  bestFitnessEver: number;
  milestone30mReached: boolean;
  trackedAgent: AgentSnapshot | null;
  agents: AgentSnapshot[];
  history: GenerationHistoryPoint[];
  networkData: NetworkInspectorData | null;
  bestGenome: GenomeData | null;
  hyperparams: EvolutionHyperparams;
}

export interface SavedCheckpointItem {
  id: string;
  mode: SimulationMode;
  name: string;
  generation: number;
  bestDistance: number;
  bestFitness: number;
  activationFn: ActivationFn;
  genome: GenomeData;
  createdAtMs: number;
}

export interface PersistedModeState {
  mode: SimulationMode;
  generation: number;
  bestDistanceEver: number;
  bestFitnessEver: number;
  milestone30mReached: boolean;
  activationFn: ActivationFn;
  populationSize: number;
  mutationRate: number;
  mutationStrength: number;
  bestGenome: GenomeData;
  history: GenerationHistoryPoint[];
  updatedAtMs: number;
}
