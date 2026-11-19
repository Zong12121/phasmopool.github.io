import type { User } from 'firebase/auth';
import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
  where,
  type Unsubscribe,
} from 'firebase/firestore';
import type {
  ActivationFn,
  EvolutionHyperparams,
  GenerationHistoryPoint,
  GenomeData,
  PersistedModeState,
  SavedCheckpointItem,
  SimulationMode,
} from '../simulation/types';
import { db, handleFirestoreError, OperationType } from './firebaseClient';

function sanitizeGenome(g: GenomeData): GenomeData {
  const clampArr = (arr: number[], maxLen: number) =>
    arr.slice(0, maxLen).map((n) => (Number.isFinite(n) ? Math.round(n * 10000) / 10000 : 0));
  const act: ActivationFn =
    g.activationFn === 'swish' || g.activationFn === 'leaky_relu' ? g.activationFn : 'tanh';
  return {
    inputSize: Math.max(1, Math.min(64, Math.round(g.inputSize))),
    hiddenSize: Math.max(1, Math.min(64, Math.round(g.hiddenSize))),
    outputSize: Math.max(1, Math.min(32, Math.round(g.outputSize))),
    activationFn: act,
    wIH: clampArr(g.wIH, 1024),
    bH: clampArr(g.bH, 64),
    wHO: clampArr(g.wHO, 512),
    bO: clampArr(g.bO, 32),
  };
}

function sanitizeHistory(history: GenerationHistoryPoint[]): GenerationHistoryPoint[] {
  return history.slice(-40).map((h) => ({
    generation: Math.max(1, Math.round(h.generation)),
    bestFitness: Number.isFinite(h.bestFitness) ? Number(h.bestFitness.toFixed(1)) : 0,
    avgFitness: Number.isFinite(h.avgFitness) ? Number(h.avgFitness.toFixed(1)) : 0,
    bestDistance: Number.isFinite(h.bestDistance) ? Number(h.bestDistance.toFixed(2)) : 0,
  }));
}

export async function ensureUserProfile(user: User): Promise<void> {
  const path = `users/${user.uid}/private/profile`;
  const ref = doc(db, 'users', user.uid, 'private', 'profile');
  try {
    const existingSnap = await getDoc(ref);
    const displayName = (user.displayName || user.email?.split('@')[0] || 'Исследователь').slice(0, 120);
    const email = (user.email || 'user@example.com').slice(0, 256);
    const photoURL = (user.photoURL || '').slice(0, 512);

    if (existingSnap.exists()) {
      await setDoc(
        ref,
        {
          uid: user.uid,
          displayName,
          email,
          ...(photoURL ? { photoURL } : {}),
          createdAt: existingSnap.data().createdAt,
          updatedAt: serverTimestamp(),
        },
        { merge: true },
      );
    } else {
      await setDoc(ref, {
        uid: user.uid,
        displayName,
        email,
        ...(photoURL ? { photoURL } : {}),
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
    }
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, path);
  }
}

export async function saveUserModeState(
  uid: string,
  mode: SimulationMode,
  params: {
    generation: number;
    bestDistanceEver: number;
    bestFitnessEver: number;
    milestone30mReached: boolean;
    hyperparams: EvolutionHyperparams;
    bestGenome: GenomeData;
    history: GenerationHistoryPoint[];
  },
): Promise<void> {
  const path = `users/${uid}/modeStates/${mode}`;
  const ref = doc(db, 'users', uid, 'modeStates', mode);
  try {
    await setDoc(ref, {
      uid,
      mode,
      generation: Math.max(1, Math.min(1000000, Math.round(params.generation))),
      bestDistanceEver: Number.isFinite(params.bestDistanceEver)
        ? Number(params.bestDistanceEver.toFixed(2))
        : 0,
      bestFitnessEver: Number.isFinite(params.bestFitnessEver)
        ? Number(params.bestFitnessEver.toFixed(1))
        : 0,
      milestone30mReached: Boolean(params.milestone30mReached),
      activationFn: params.hyperparams.activationFn,
      populationSize: Math.max(4, Math.min(32, Math.round(params.hyperparams.populationSize))),
      mutationRate: Math.max(0.01, Math.min(1.0, Number(params.hyperparams.mutationRate.toFixed(3)))),
      mutationStrength: Math.max(0.01, Math.min(2.0, Number(params.hyperparams.mutationStrength.toFixed(3)))),
      bestGenome: sanitizeGenome(params.bestGenome),
      history: sanitizeHistory(params.history),
      updatedAt: serverTimestamp(),
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.WRITE, path);
  }
}

export async function loadUserModeState(
  uid: string,
  mode: SimulationMode,
): Promise<PersistedModeState | null> {
  const path = `users/${uid}/modeStates/${mode}`;
  const ref = doc(db, 'users', uid, 'modeStates', mode);
  try {
    const snap = await getDoc(ref);
    if (!snap.exists()) return null;
    const d = snap.data();
    return {
      mode: d.mode as SimulationMode,
      generation: Number(d.generation) || 1,
      bestDistanceEver: Number(d.bestDistanceEver) || 0,
      bestFitnessEver: Number(d.bestFitnessEver) || 0,
      milestone30mReached: Boolean(d.milestone30mReached),
      activationFn: (d.activationFn as ActivationFn) || 'tanh',
      populationSize: Number(d.populationSize) || 12,
      mutationRate: Number(d.mutationRate) || 0.16,
      mutationStrength: Number(d.mutationStrength) || 0.22,
      bestGenome: d.bestGenome as GenomeData,
      history: Array.isArray(d.history) ? (d.history as GenerationHistoryPoint[]) : [],
      updatedAtMs: d.updatedAt?.toMillis?.() ?? Date.now(),
    };
  } catch (error) {
    handleFirestoreError(error, OperationType.GET, path);
  }
}

export async function listUserModeStates(uid: string): Promise<PersistedModeState[]> {
  const path = `users/${uid}/modeStates`;
  try {
    const q = query(collection(db, 'users', uid, 'modeStates'), where('uid', '==', uid));
    const snap = await getDocs(q);
    return snap.docs.map((docSnap) => {
      const d = docSnap.data();
      return {
        mode: d.mode as SimulationMode,
        generation: Number(d.generation) || 1,
        bestDistanceEver: Number(d.bestDistanceEver) || 0,
        bestFitnessEver: Number(d.bestFitnessEver) || 0,
        milestone30mReached: Boolean(d.milestone30mReached),
        activationFn: (d.activationFn as ActivationFn) || 'tanh',
        populationSize: Number(d.populationSize) || 12,
        mutationRate: Number(d.mutationRate) || 0.16,
        mutationStrength: Number(d.mutationStrength) || 0.22,
        bestGenome: d.bestGenome as GenomeData,
        history: Array.isArray(d.history) ? (d.history as GenerationHistoryPoint[]) : [],
        updatedAtMs: d.updatedAt?.toMillis?.() ?? Date.now(),
      };
    });
  } catch (error) {
    handleFirestoreError(error, OperationType.LIST, path);
  }
}

export async function saveUserCheckpoint(
  uid: string,
  params: {
    mode: SimulationMode;
    name: string;
    generation: number;
    bestDistance: number;
    bestFitness: number;
    activationFn: ActivationFn;
    genome: GenomeData;
  },
): Promise<string> {
  const checkpointId = `cp_${params.mode}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const path = `users/${uid}/checkpoints/${checkpointId}`;
  const ref = doc(db, 'users', uid, 'checkpoints', checkpointId);
  try {
    const cleanName = params.name.trim().slice(0, 80) || `${params.mode} Gen ${params.generation}`;
    await setDoc(ref, {
      uid,
      mode: params.mode,
      name: cleanName,
      generation: Math.max(1, Math.min(1000000, Math.round(params.generation))),
      bestDistance: Number.isFinite(params.bestDistance) ? Number(params.bestDistance.toFixed(2)) : 0,
      bestFitness: Number.isFinite(params.bestFitness) ? Number(params.bestFitness.toFixed(1)) : 0,
      activationFn: params.activationFn,
      genome: sanitizeGenome(params.genome),
      createdAt: serverTimestamp(),
    });
    return checkpointId;
  } catch (error) {
    handleFirestoreError(error, OperationType.CREATE, path);
  }
}

export function subscribeUserCheckpoints(
  uid: string,
  onData: (items: SavedCheckpointItem[]) => void,
): Unsubscribe {
  const path = `users/${uid}/checkpoints`;
  const q = query(
    collection(db, 'users', uid, 'checkpoints'),
    where('uid', '==', uid),
    orderBy('createdAt', 'desc'),
  );
  return onSnapshot(
    q,
    (snap) => {
      const items: SavedCheckpointItem[] = snap.docs.map((docSnap) => {
        const d = docSnap.data();
        return {
          id: docSnap.id,
          mode: d.mode as SimulationMode,
          name: String(d.name ?? 'Чекпоинт'),
          generation: Number(d.generation) || 1,
          bestDistance: Number(d.bestDistance) || 0,
          bestFitness: Number(d.bestFitness) || 0,
          activationFn: (d.activationFn as ActivationFn) || 'tanh',
          genome: d.genome as GenomeData,
          createdAtMs: d.createdAt?.toMillis?.() ?? Date.now(),
        };
      });
      onData(items);
    },
    (error) => {
      handleFirestoreError(error, OperationType.LIST, path);
    },
  );
}

export async function deleteUserCheckpoint(uid: string, checkpointId: string): Promise<void> {
  const path = `users/${uid}/checkpoints/${checkpointId}`;
  try {
    await deleteDoc(doc(db, 'users', uid, 'checkpoints', checkpointId));
  } catch (error) {
    handleFirestoreError(error, OperationType.DELETE, path);
  }
}
