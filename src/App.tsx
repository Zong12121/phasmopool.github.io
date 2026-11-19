import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  onAuthStateChanged,
  type User,
} from 'firebase/auth';
import {
  Activity,
  BookmarkPlus,
  Car,
  ChevronDown,
  ChevronUp,
  CircleDot,
  Cloud,
  Database,
  Eye,
  FastForward,
  Footprints,
  Gauge,
  LogIn,
  LogOut,
  Mountain,
  Pause,
  Plane,
  Play,
  RotateCcw,
  SkipForward,
  Sliders,
  Sparkles,
  Target,
  Trash2,
  Wind,
} from 'lucide-react';
import { FitnessChart } from './components/FitnessChart';
import { NeuralNetVisualizer } from './components/NeuralNetVisualizer';
import {
  auth,
  signInWithGoogle,
  signOutUser,
  testFirestoreConnection,
} from './firebase/firebaseClient';
import {
  deleteUserCheckpoint,
  ensureUserProfile,
  listUserModeStates,
  loadUserModeState,
  saveUserCheckpoint,
  saveUserModeState,
  subscribeUserCheckpoints,
} from './firebase/storageService';
import {
  SimulationEngine,
  type PersistedGenerationPayload,
} from './simulation/SimulationEngine';
import type {
  ActivationFn,
  EnvironmentChallenges,
  EvolutionHyperparams,
  PersistedModeState,
  SavedCheckpointItem,
  SimulationMode,
  SimulationTelemetry,
} from './simulation/types';

type DockTab = 'controls' | 'lab' | 'network' | 'chart' | 'database';

const MODE_TABS: Array<{ id: SimulationMode; label: string }> = [
  { id: 'navigation', label: '1. Навигация' },
  { id: 'balance', label: '2. Балансировка' },
  { id: 'parkour', label: '3. Препятствия' },
  { id: 'biomechanics', label: '4. Прямохождение' },
  { id: 'manual_car', label: '5. Авто МКПП' },
  { id: 'drone', label: '6. Квадрокоптер' },
];

const MODE_NAMES: Record<SimulationMode, string> = {
  navigation: 'Навигация',
  balance: 'Балансировка',
  parkour: 'Препятствия',
  biomechanics: 'Прямохождение',
  manual_car: 'Авто МКПП',
  drone: 'Квадрокоптер',
};

function formatGear(g?: number): string {
  if (g === undefined) return 'N';
  if (g === -1) return 'R';
  if (g === 0) return 'N';
  return String(g);
}

export default function App() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<SimulationEngine | null>(null);

  const [mode, setMode] = useState<SimulationMode>('biomechanics');
  const [paused, setPaused] = useState(false);
  const [simSpeed, setSimSpeed] = useState(1);
  const [followLeader, setFollowLeader] = useState(true);
  const [dockCollapsed, setDockCollapsed] = useState(false);
  const [dockTab, setDockTab] = useState<DockTab>('controls');
  const [contextLost, setContextLost] = useState(false);

  const [challenges, setChallenges] = useState<EnvironmentChallenges>({
    unevenTerrain: false,
    headwind: false,
    windForce: 22,
    ballsThrown: 0,
    manualUnlock30m: false,
  });

  const [telemetry, setTelemetry] = useState<SimulationTelemetry | null>(null);

  // Auth & Per-User Cloud Database State
  const [user, setUser] = useState<User | null>(null);
  const userRef = useRef<User | null>(null);
  const [authBusy, setAuthBusy] = useState(false);
  const [syncStatus, setSyncStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [savedModeStates, setSavedModeStates] = useState<PersistedModeState[]>([]);
  const [checkpoints, setCheckpoints] = useState<SavedCheckpointItem[]>([]);
  const [checkpointName, setCheckpointName] = useState('');
  const [savingCheckpoint, setSavingCheckpoint] = useState(false);

  const persistGeneration = useCallback(async (payload: PersistedGenerationPayload) => {
    const activeUser = userRef.current;
    if (!activeUser) return;
    setSyncStatus('saving');
    try {
      await saveUserModeState(activeUser.uid, payload.mode, {
        generation: payload.generation,
        bestDistanceEver: payload.bestDistanceEver,
        bestFitnessEver: payload.bestFitnessEver,
        milestone30mReached: payload.milestone30mReached,
        hyperparams: payload.hyperparams,
        bestGenome: payload.bestGenome,
        history: payload.history,
      });
      setSyncStatus('saved');
      const states = await listUserModeStates(activeUser.uid);
      setSavedModeStates(states);
    } catch {
      setSyncStatus('error');
    }
  }, []);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const engine = new SimulationEngine(
      container,
      (data) => {
        setTelemetry(data);
      },
      (lost) => setContextLost(lost),
      (payload) => {
        void persistGeneration(payload);
      },
    );
    engineRef.current = engine;

    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if (e.key === 'h' || e.key === 'H' || e.key === 'р' || e.key === 'Р') {
        setDockCollapsed((prev) => !prev);
      } else if (e.key === ' ') {
        e.preventDefault();
        setPaused((prev) => {
          const next = !prev;
          engine.setPaused(next);
          return next;
        });
      }
    };
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      engine.dispose();
      engineRef.current = null;
    };
  }, [persistGeneration]);

  // Firebase Auth & Firestore Checkpoints Subscription
  useEffect(() => {
    void testFirestoreConnection();
    let unsubCheckpoints: (() => void) | null = null;
    const unsubAuth = onAuthStateChanged(auth, async (nextUser) => {
      userRef.current = nextUser;
      setUser(nextUser);

      if (unsubCheckpoints) {
        unsubCheckpoints();
        unsubCheckpoints = null;
      }

      if (!nextUser) {
        setSavedModeStates([]);
        setCheckpoints([]);
        setSyncStatus('idle');
        return;
      }

      await ensureUserProfile(nextUser);
      const states = await listUserModeStates(nextUser.uid);
      setSavedModeStates(states);

      // Restore current mode progress from database if it exists and local is fresh Gen 1
      const currentSaved = await loadUserModeState(nextUser.uid, mode);
      if (currentSaved && currentSaved.generation > 1 && engineRef.current) {
        engineRef.current.applyPersistedModeState(currentSaved);
      }

      unsubCheckpoints = subscribeUserCheckpoints(nextUser.uid, (items) => {
        setCheckpoints(items);
      });
    });

    return () => {
      unsubAuth();
      if (unsubCheckpoints) unsubCheckpoints();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleModeChange = async (nextMode: SimulationMode) => {
    setMode(nextMode);
    engineRef.current?.setMode(nextMode);

    if (userRef.current) {
      try {
        const saved = await loadUserModeState(userRef.current.uid, nextMode);
        if (saved && saved.bestGenome && engineRef.current) {
          engineRef.current.applyPersistedModeState(saved);
        }
      } catch {
        // Ignore transient lookup issues
      }
    }
  };

  const handleGoogleSignIn = async () => {
    setAuthBusy(true);
    try {
      await signInWithGoogle();
    } finally {
      setAuthBusy(false);
    }
  };

  const handleSignOut = async () => {
    setAuthBusy(true);
    try {
      await signOutUser();
    } finally {
      setAuthBusy(false);
    }
  };

  const handleSaveNow = async () => {
    if (!user || !engineRef.current) return;
    const payload = engineRef.current.getExportableState();
    if (!payload) return;
    await persistGeneration(payload);
  };

  const handleSaveCheckpoint = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !engineRef.current) return;
    const payload = engineRef.current.getExportableState();
    if (!payload) return;
    setSavingCheckpoint(true);
    try {
      const label =
        checkpointName.trim() ||
        `${MODE_NAMES[payload.mode]} · Пок. ${payload.generation} (${payload.bestDistanceEver.toFixed(1)}м)`;
      await saveUserCheckpoint(user.uid, {
        mode: payload.mode,
        name: label,
        generation: payload.generation,
        bestDistance: payload.bestDistanceEver,
        bestFitness: payload.bestFitnessEver,
        activationFn: payload.hyperparams.activationFn,
        genome: payload.bestGenome,
      });
      await persistGeneration(payload);
      setCheckpointName('');
    } finally {
      setSavingCheckpoint(false);
    }
  };

  const handleLoadCheckpoint = (cp: SavedCheckpointItem) => {
    setMode(cp.mode);
    engineRef.current?.loadCheckpointGenome(cp.mode, cp.genome, cp.generation, cp.activationFn);
  };

  const handleLoadModeState = (st: PersistedModeState) => {
    setMode(st.mode);
    engineRef.current?.applyPersistedModeState(st);
  };

  const togglePause = () => {
    const next = !paused;
    setPaused(next);
    engineRef.current?.setPaused(next);
  };

  const handleSpeedChange = (s: number) => {
    setSimSpeed(s);
    engineRef.current?.setSimSpeed(s);
  };

  const handleToggleFollow = () => {
    const next = !followLeader;
    setFollowLeader(next);
    engineRef.current?.setFollowLeader(next);
  };

  const updateChallenge = (patch: Partial<EnvironmentChallenges>) => {
    setChallenges((prev) => {
      const next = { ...prev, ...patch };
      engineRef.current?.setChallenges(next);
      return next;
    });
  };

  const updateHyperparams = (patch: Partial<EvolutionHyperparams>) => {
    engineRef.current?.setHyperparams(patch);
  };

  const isBiomechanics = mode === 'biomechanics';
  const isManualCar = mode === 'manual_car';
  const isDrone = mode === 'drone';
  const bestRecord = telemetry?.bestDistanceEver ?? 0;
  const challengesUnlocked =
    (telemetry?.milestone30mReached ?? false) || bestRecord >= 30 || challenges.manualUnlock30m;
  const tracked = telemetry?.trackedAgent ?? null;
  const hp = telemetry?.hyperparams;

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#080b10] text-slate-100 select-none">
      {/* Isolated 3D WebGL Mount Container */}
      <div ref={mountRef} className="absolute inset-0 h-full w-full" />

      {/* Context Lost Fallback */}
      {contextLost && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/95 p-6">
          <div className="max-w-md rounded-lg border border-slate-800 bg-slate-900 p-6 text-center">
            <h2 className="font-display text-lg font-semibold text-amber-400">
              Восстановление WebGL-контекста
            </h2>
            <p className="mt-2 text-sm text-slate-300">
              Графический контекст был временно приостановлен. Перезапустите сцену или обновите страницу.
            </p>
          </div>
        </div>
      )}

      {/* Top Navigation Bar (Clean 3-Zone Contract) */}
      <header className="pointer-events-auto relative z-20 flex items-center justify-between gap-3 border-b border-slate-800/80 bg-[#0b1018]/92 px-4 py-2 overflow-x-auto">
        <span className="font-display text-base font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0">
          Биомеханика 3D
        </span>

        <nav className="flex items-center gap-1 rounded-lg bg-slate-900/90 p-1 border border-slate-800 shrink-0">
          {MODE_TABS.map((tab) => {
            const active = mode === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => handleModeChange(tab.id)}
                className={`px-2.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  active
                    ? 'bg-amber-500 text-slate-950 font-semibold'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/70'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={togglePause}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-slate-100 hover:bg-slate-800 transition-colors whitespace-nowrap cursor-pointer"
          >
            {paused ? <Play className="h-3.5 w-3.5 text-amber-400" /> : <Pause className="h-3.5 w-3.5 text-sky-400" />}
            <span>{paused ? 'Старт' : 'Пауза'}</span>
          </button>

          <button
            type="button"
            onClick={() => engineRef.current?.nextGenerationNow()}
            title="Завершить поколение и перейти к следующему"
            className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-800 transition-colors whitespace-nowrap cursor-pointer"
          >
            <SkipForward className="h-3.5 w-3.5 text-slate-400" />
            <span>Пок. +1</span>
          </button>

          <button
            type="button"
            onClick={() => engineRef.current?.resetEvolution()}
            title="Сбросить эволюцию к 1 поколению"
            className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-800 transition-colors whitespace-nowrap cursor-pointer"
          >
            <RotateCcw className="h-3.5 w-3.5 text-slate-400" />
            <span>Сброс</span>
          </button>

          {user ? (
            <div className="ml-1 flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/90 px-2 py-1">
              <Cloud
                className={`h-3.5 w-3.5 ${
                  syncStatus === 'saving'
                    ? 'text-amber-400'
                    : syncStatus === 'error'
                      ? 'text-rose-400'
                      : 'text-emerald-400'
                }`}
              />
              <span className="text-[11px] font-medium text-slate-200 max-w-[96px] truncate">
                {user.displayName?.split(' ')[0] || 'Профиль'}
              </span>
              <button
                type="button"
                onClick={handleSignOut}
                disabled={authBusy}
                title="Выйти из аккаунта"
                className="text-slate-400 hover:text-white cursor-pointer"
              >
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={handleGoogleSignIn}
              disabled={authBusy}
              className="ml-1 inline-flex items-center gap-1.5 rounded-md border border-sky-500/50 bg-sky-500/15 px-2.5 py-1.5 text-xs font-medium text-sky-300 hover:bg-sky-500/25 transition-colors whitespace-nowrap cursor-pointer"
            >
              <LogIn className="h-3.5 w-3.5" />
              <span>Войти (БД)</span>
            </button>
          )}
        </div>
      </header>

      {/* Top-Left Compact Live Telemetry Strip */}
      <div className="pointer-events-none absolute top-14 left-4 z-10 flex flex-col gap-2 max-w-[355px]">
        <div className="pointer-events-auto rounded-lg border border-slate-800/80 bg-[#0b1018]/92 px-3.5 py-2.5">
          <div className="flex items-center justify-between gap-3 text-xs text-slate-400">
            <span>
              Поколение <strong className="font-mono-tabular text-slate-100">{telemetry?.generation ?? 1}</strong>
            </span>
            <span aria-hidden="true">·</span>
            <span>
              В строю{' '}
              <strong className="font-mono-tabular text-sky-400">
                {telemetry?.aliveCount ?? 0}/{telemetry?.totalCount ?? 0}
              </strong>
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-mono-tabular text-slate-300">
              {(telemetry?.elapsedTime ?? 0).toFixed(1)}с / {telemetry?.generationDuration ?? 28}с
            </span>
          </div>

          <div className="mt-2 grid grid-cols-3 gap-3 border-t border-slate-800/70 pt-2">
            <div>
              <div className="text-[11px] text-slate-400">Дистанция</div>
              <div className="font-mono-tabular text-sm font-semibold text-amber-400">
                {(tracked?.distance ?? 0).toFixed(2)} м
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">Рекорд</div>
              <div className="font-mono-tabular text-sm font-semibold text-slate-100">
                {bestRecord.toFixed(2)} м
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">Скорость</div>
              <div className="font-mono-tabular text-sm font-semibold text-sky-400">
                {isManualCar
                  ? `${((tracked?.speed ?? 0) * 3.6).toFixed(0)} км/ч`
                  : `${(tracked?.speed ?? 0).toFixed(2)} м/с`}
              </div>
            </div>
          </div>

          {isBiomechanics && tracked && (
            <div className="mt-2 flex items-center justify-between gap-2 border-t border-slate-800/70 pt-2 text-[11px] text-slate-400">
              <span>
                Высота: <strong className="font-mono-tabular text-slate-200">{tracked.torsoHeight.toFixed(2)} м</strong>
              </span>
              <span>
                Тангаж: <strong className="font-mono-tabular text-slate-200">{tracked.pitchDeg.toFixed(0)}°</strong>
              </span>
              <span className="flex items-center gap-1">
                Стопы:
                <span className={tracked.leftContact ? 'text-amber-400 font-semibold' : 'text-slate-600'}>Л</span>
                <span className={tracked.rightContact ? 'text-amber-400 font-semibold' : 'text-slate-600'}>П</span>
              </span>
            </div>
          )}

          {/* Mode 5: Manual Transmission Tachometer & Pedals HUD */}
          {isManualCar && tracked && (
            <div className="mt-2 border-t border-slate-800/70 pt-2">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-md border border-amber-500/60 bg-amber-500/15 font-display text-base font-bold text-amber-300">
                    {formatGear(tracked.gear)}
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-400">МКПП</div>
                    <div className="font-mono-tabular text-xs font-semibold text-slate-100">
                      {tracked.rpm ?? 900} RPM
                    </div>
                  </div>
                </div>

                <div className="flex-1">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full transition-all duration-75 ${
                        (tracked.rpm ?? 900) > 6400
                          ? 'bg-rose-500'
                          : (tracked.rpm ?? 900) > 4500
                            ? 'bg-amber-400'
                            : 'bg-sky-400'
                      }`}
                      style={{
                        width: `${Math.min(100, Math.max(8, (((tracked.rpm ?? 900) - 400) / 6800) * 100))}%`,
                      }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between text-[10px] font-mono-tabular text-slate-400">
                    <span>Газ {Math.round((tracked.throttle ?? 0) * 100)}%</span>
                    <span>Торм {Math.round((tracked.brake ?? 0) * 100)}%</span>
                    <span>Сцепл {Math.round((tracked.clutch ?? 0) * 100)}%</span>
                  </div>
                </div>
              </div>
              {tracked.stalled && (
                <div className="mt-1.5 rounded bg-rose-500/20 px-2 py-0.5 text-center text-[11px] font-semibold text-rose-300">
                  Двигатель заглох
                </div>
              )}
            </div>
          )}

          {/* Mode 6: Drone Checkpoints HUD */}
          {isDrone && tracked && (
            <div className="mt-2 flex items-center justify-between gap-2 border-t border-slate-800/70 pt-2 text-[11px] text-slate-400">
              <span>
                Кольца: <strong className="font-mono-tabular text-emerald-400">{tracked.checkpointsPassed ?? 0}</strong>
              </span>
              <span>
                Высота: <strong className="font-mono-tabular text-slate-200">{tracked.torsoHeight.toFixed(2)} м</strong>
              </span>
              <span>
                Крен/Тангаж:{' '}
                <strong className="font-mono-tabular text-slate-200">
                  {tracked.rollDeg.toFixed(0)}°/{tracked.pitchDeg.toFixed(0)}°
                </strong>
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Top-Right Compact Camera & Mode Info Strip */}
      <div className="pointer-events-auto absolute top-14 right-4 z-10 flex items-center gap-2">
        <button
          type="button"
          onClick={handleToggleFollow}
          className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
            followLeader
              ? 'border-sky-500/50 bg-[#0b1018]/92 text-sky-300'
              : 'border-slate-800 bg-[#0b1018]/92 text-slate-300 hover:text-white'
          }`}
        >
          <Eye className="h-3.5 w-3.5" />
          <span>{followLeader ? 'Камера: лидер' : 'Свободная камера'}</span>
        </button>
      </div>

      {/* Collapsible Bottom Control Dock */}
      <div
        className={`pointer-events-auto fixed left-1/2 z-20 -translate-x-1/2 transition-transform duration-200 ease-out ${
          dockCollapsed ? 'translate-y-[calc(100%-36px)]' : 'translate-y-0'
        }`}
        style={{ bottom: '12px', width: 'min(1020px, 96vw)' }}
      >
        {/* Collapse Toggle Handle */}
        <div className="flex justify-center">
          <button
            type="button"
            onClick={() => setDockCollapsed((v) => !v)}
            className="inline-flex items-center gap-1.5 rounded-t-lg border border-b-0 border-slate-800 bg-[#0c121b]/95 px-4 py-1 text-[11px] font-medium text-slate-300 hover:text-white cursor-pointer"
          >
            {dockCollapsed ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            <span>{dockCollapsed ? 'Показать панель управления (H)' : 'Скрыть панель (H)'}</span>
          </button>
        </div>

        {/* Main Dock Surface */}
        <div className="rounded-xl border border-slate-800 bg-[#0c121b]/96 p-3.5 shadow-2xl">
          {/* Dock Sub-navigation & Speed Controls */}
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
            <div className="flex flex-wrap items-center gap-1 rounded-lg bg-slate-900 p-1 border border-slate-800">
              <button
                type="button"
                onClick={() => setDockTab('controls')}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  dockTab === 'controls' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Footprints className="h-3.5 w-3.5" />
                <span>Трасса и пресеты</span>
              </button>
              <button
                type="button"
                onClick={() => setDockTab('lab')}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  dockTab === 'lab' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sliders className="h-3.5 w-3.5" />
                <span>Лаборатория ИИ</span>
              </button>
              <button
                type="button"
                onClick={() => setDockTab('network')}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  dockTab === 'network' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Activity className="h-3.5 w-3.5" />
                <span>
                  Нейросеть ({telemetry?.networkData?.inputs.length ?? 18}→
                  {telemetry?.networkData?.outputs.length ?? 6})
                </span>
              </button>
              <button
                type="button"
                onClick={() => setDockTab('chart')}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  dockTab === 'chart' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Target className="h-3.5 w-3.5" />
                <span>Прогресс</span>
              </button>
              <button
                type="button"
                onClick={() => setDockTab('database')}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                  dockTab === 'database' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Database className="h-3.5 w-3.5" />
                <span>База ({checkpoints.length})</span>
              </button>
            </div>

            {/* Simulation Speed Multiplier */}
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1 text-xs text-slate-400">
                <FastForward className="h-3.5 w-3.5" />
                <span>Скорость:</span>
              </span>
              <div className="flex items-center gap-1 rounded-lg bg-slate-900 p-1 border border-slate-800">
                {[1, 2, 5, 10].map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => handleSpeedChange(s)}
                    className={`px-2 py-1 text-xs font-mono-tabular rounded transition-colors cursor-pointer ${
                      simSpeed === s
                        ? 'bg-amber-500 text-slate-950 font-semibold'
                        : 'text-slate-300 hover:text-white'
                    }`}
                  >
                    {s}×
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* TAB 1: CONTROLS & PRESETS */}
          {dockTab === 'controls' && (
            <div className="mt-3 flex flex-col gap-3">
              {isBiomechanics && (
                <>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-3 text-xs text-slate-300">
                      <span className="font-display font-semibold text-amber-400">
                        Биомеханика прямохождения
                      </span>
                      <span aria-hidden="true" className="text-slate-600">·</span>
                      <span className="text-slate-400">
                        9 тел cannon-es · 18 сенсоров · 6 моторизированных шарниров
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={() => engineRef.current?.loadPretrainedBipedGait()}
                      className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 text-xs font-medium text-amber-300 hover:bg-amber-500/25 transition-colors whitespace-nowrap cursor-pointer"
                    >
                      <Sparkles className="h-3.5 w-3.5" />
                      <span>Загрузить обученную походку</span>
                    </button>
                  </div>

                  {/* Environmental Challenges Section */}
                  <div className="rounded-lg border border-slate-800 bg-slate-900/65 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                      <div className="flex items-center gap-2 text-xs">
                        <span className="font-semibold text-slate-200">
                          Усложнение среды (после порога 30 м)
                        </span>
                        <span aria-hidden="true" className="text-slate-600">·</span>
                        {challengesUnlocked ? (
                          <span className="text-emerald-400 font-medium">Испытания активны</span>
                        ) : (
                          <span className="text-slate-400 font-mono-tabular">
                            Прогресс: {Math.min(30, bestRecord).toFixed(1)} / 30.0 м
                          </span>
                        )}
                      </div>

                      {!challengesUnlocked && (
                        <button
                          type="button"
                          onClick={() => updateChallenge({ manualUnlock30m: true })}
                          className="text-[11px] font-medium text-sky-400 hover:text-sky-300 underline underline-offset-2 cursor-pointer"
                        >
                          Разблокировать испытания сейчас
                        </button>
                      )}
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                      <button
                        type="button"
                        disabled={!challengesUnlocked}
                        onClick={() => updateChallenge({ unevenTerrain: !challenges.unevenTerrain })}
                        className={`flex items-center justify-between gap-2 rounded-lg border px-3.5 py-2.5 text-xs font-medium transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                          challenges.unevenTerrain
                            ? 'border-amber-500 bg-amber-500/15 text-amber-200'
                            : 'border-slate-700 bg-slate-900 text-slate-200 hover:bg-slate-800'
                        }`}
                      >
                        <span className="inline-flex items-center gap-2">
                          <Mountain className="h-4 w-4 text-amber-400" />
                          <span>Неровный рельеф / Ступеньки</span>
                        </span>
                        <span className="font-mono-tabular text-[11px]">
                          {challenges.unevenTerrain ? 'ВКЛ' : 'ВЫКЛ'}
                        </span>
                      </button>

                      <div
                        className={`flex items-center justify-between gap-2 rounded-lg border px-3.5 py-2 text-xs font-medium transition-colors ${
                          !challengesUnlocked ? 'opacity-40' : ''
                        } ${
                          challenges.headwind
                            ? 'border-sky-500 bg-sky-500/15 text-sky-200'
                            : 'border-slate-700 bg-slate-900 text-slate-200'
                        }`}
                      >
                        <button
                          type="button"
                          disabled={!challengesUnlocked}
                          onClick={() => updateChallenge({ headwind: !challenges.headwind })}
                          className="inline-flex items-center gap-2 cursor-pointer disabled:cursor-not-allowed whitespace-nowrap"
                        >
                          <Wind className="h-4 w-4 text-sky-400" />
                          <span>Встречный ветер</span>
                        </button>
                        <div className="flex items-center gap-2">
                          <input
                            type="range"
                            min={10}
                            max={45}
                            step={5}
                            disabled={!challengesUnlocked}
                            value={challenges.windForce}
                            onChange={(e) => updateChallenge({ windForce: Number(e.target.value) })}
                            className="w-16 accent-sky-400 cursor-pointer"
                            aria-label="Сила встречного ветра"
                          />
                          <span className="font-mono-tabular text-[11px] w-8 text-right">
                            {challenges.headwind ? `${challenges.windForce}Н` : 'ВЫКЛ'}
                          </span>
                        </div>
                      </div>

                      <button
                        type="button"
                        disabled={!challengesUnlocked}
                        onClick={() => engineRef.current?.throwBallAtLeader()}
                        className="flex items-center justify-between gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3.5 py-2.5 text-xs font-medium text-rose-200 hover:bg-rose-500/20 transition-colors whitespace-nowrap cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        <span className="inline-flex items-center gap-2">
                          <CircleDot className="h-4 w-4 text-rose-400" />
                          <span>Бросок мяча</span>
                        </span>
                        <span className="font-mono-tabular text-[11px] text-rose-300">
                          Выстрел ({challenges.ballsThrown})
                        </span>
                      </button>
                    </div>
                  </div>
                </>
              )}

              {mode === 'balance' && (
                <div className="flex flex-wrap items-center justify-between gap-4 py-1 text-xs text-slate-300">
                  <div>
                    <div className="font-display font-semibold text-sky-400 mb-0.5">
                      Умная балансировка на рельсе с порывами ветра
                    </div>
                    <p className="text-slate-400">
                      Тележки активно разгоняются на подшипниковых рельсах, компенсируя наклоны и периодические порывы ветра.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => engineRef.current?.loadPretrainedCartPole()}
                    className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 text-xs font-medium text-amber-300 hover:bg-amber-500/25 transition-colors whitespace-nowrap cursor-pointer"
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    <span>Загрузить обученный баланс</span>
                  </button>
                </div>
              )}

              {mode === 'manual_car' && (
                <div className="flex flex-wrap items-center justify-between gap-4 py-1 text-xs text-slate-300">
                  <div className="max-w-xl">
                    <div className="flex items-center gap-2 font-display font-semibold text-amber-400 mb-0.5">
                      <Car className="h-4 w-4" />
                      <span>Вождение автомобиля на механической КПП (N / 1–5)</span>
                    </div>
                    <p className="text-slate-400">
                      Нейросеть управляет рулём, газом, тормозом, сцеплением и рычагом КПП. При резком броске сцепления на низких оборотах двигатель глохнет.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => engineRef.current?.loadPretrainedManualCar()}
                    className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 text-xs font-medium text-amber-300 hover:bg-amber-500/25 transition-colors whitespace-nowrap cursor-pointer"
                  >
                    <Gauge className="h-3.5 w-3.5" />
                    <span>Загрузить обученного водителя МКПП</span>
                  </button>
                </div>
              )}

              {mode === 'drone' && (
                <div className="flex flex-wrap items-center justify-between gap-4 py-1 text-xs text-slate-300">
                  <div className="max-w-xl">
                    <div className="flex items-center gap-2 font-display font-semibold text-sky-400 mb-0.5">
                      <Plane className="h-4 w-4" />
                      <span>Автопилот 3D-квадрокоптера (4 независимых винта)</span>
                    </div>
                    <p className="text-slate-400">
                      ИИ регулирует тягу каждого из 4 роторов (FL, FR, RL, RR), удерживая тангаж/крен и пролетая сквозь воздушные кольца.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => engineRef.current?.loadPretrainedDrone()}
                    className="inline-flex items-center gap-1.5 rounded-md border border-sky-500/40 bg-sky-500/15 px-3 py-1.5 text-xs font-medium text-sky-300 hover:bg-sky-500/25 transition-colors whitespace-nowrap cursor-pointer"
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    <span>Загрузить автопилот дрона</span>
                  </button>
                </div>
              )}

              {(mode === 'navigation' || mode === 'parkour') && (
                <div className="flex flex-wrap items-center justify-between gap-4 py-1 text-xs text-slate-300">
                  <div>
                    {mode === 'navigation' &&
                      'Автономный ровер с 5 лучевыми дальномерами обучается объезжать препятствия и собирать маяки.'}
                    {mode === 'parkour' &&
                      'Скоростной прыгающий агент с физикой cannon-es учится перепрыгивать барьеры на трассе.'}
                  </div>
                  <button
                    type="button"
                    onClick={() => handleModeChange('manual_car')}
                    className="rounded-md bg-amber-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 hover:bg-amber-400 transition-colors whitespace-nowrap cursor-pointer"
                  >
                    Перейти к Режиму 5: Авто МКПП
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: AI EVOLUTION LAB (HYPERPARAMETERS + ACTIVATIONS + HALL OF FAME) */}
          {dockTab === 'lab' && hp && (
            <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-2.5">
                <div className="text-slate-400 mb-1.5">Функция активации</div>
                <div className="grid grid-cols-3 gap-1">
                  {(['tanh', 'swish', 'leaky_relu'] as ActivationFn[]).map((fn) => (
                    <button
                      key={fn}
                      type="button"
                      onClick={() => updateHyperparams({ activationFn: fn })}
                      className={`rounded px-1.5 py-1 font-mono text-[11px] cursor-pointer transition-colors ${
                        hp.activationFn === fn
                          ? 'bg-amber-500 text-slate-950 font-semibold'
                          : 'bg-slate-800 text-slate-300 hover:text-white'
                      }`}
                    >
                      {fn === 'leaky_relu' ? 'LReLU' : fn}
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-2.5">
                <div className="flex justify-between text-slate-400 mb-1.5">
                  <span>Размер популяции</span>
                  <strong className="font-mono-tabular text-slate-200">{hp.populationSize}</strong>
                </div>
                <div className="grid grid-cols-4 gap-1">
                  {[8, 12, 16, 20].map((sz) => (
                    <button
                      key={sz}
                      type="button"
                      onClick={() => updateHyperparams({ populationSize: sz })}
                      className={`rounded py-1 font-mono text-[11px] cursor-pointer ${
                        hp.populationSize === sz
                          ? 'bg-sky-500 text-slate-950 font-semibold'
                          : 'bg-slate-800 text-slate-300 hover:text-white'
                      }`}
                    >
                      {sz}
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-2.5">
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Шанс мутации</span>
                  <strong className="font-mono-tabular text-amber-300">
                    {Math.round(hp.mutationRate * 100)}%
                  </strong>
                </div>
                <input
                  type="range"
                  min={0.04}
                  max={0.45}
                  step={0.02}
                  value={hp.mutationRate}
                  onChange={(e) => updateHyperparams({ mutationRate: Number(e.target.value) })}
                  className="w-full accent-amber-400 cursor-pointer"
                />
                <div className="flex justify-between text-slate-400 mt-1 mb-0.5">
                  <span>Сила мутации (σ)</span>
                  <strong className="font-mono-tabular text-sky-300">{hp.mutationStrength.toFixed(2)}</strong>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={0.6}
                  step={0.03}
                  value={hp.mutationStrength}
                  onChange={(e) => updateHyperparams({ mutationStrength: Number(e.target.value) })}
                  className="w-full accent-sky-400 cursor-pointer"
                />
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-2.5 flex flex-col justify-between">
                <div>
                  <div className="text-slate-400 mb-1">Зал славы (Hall of Fame)</div>
                  <p className="text-[11px] text-slate-500 leading-snug">
                    Гарантирует перенос лучшего генома за всю историю в следующее поколение.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => updateHyperparams({ hallOfFame: !hp.hallOfFame })}
                  className={`mt-2 w-full rounded py-1.5 text-xs font-medium cursor-pointer transition-colors ${
                    hp.hallOfFame
                      ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-300'
                      : 'bg-slate-800 border border-slate-700 text-slate-400'
                  }`}
                >
                  {hp.hallOfFame ? 'Элитизм зала славы: ВКЛ' : 'Элитизм зала славы: ВЫКЛ'}
                </button>
              </div>
            </div>
          )}

          {/* TAB 3: NEURAL NET INSPECTOR */}
          {dockTab === 'network' && (
            <div className="mt-3">
              <div className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
                <span>
                  Активная нейросеть лидера #{tracked?.id ?? 0} (Янтарный: +активация · Голубой: −активация)
                </span>
                <span className="font-mono-tabular text-slate-300">
                  Входы: {telemetry?.networkData?.inputs.length ?? 0} · Скрытый слой:{' '}
                  {telemetry?.networkData?.hidden.length ?? 0} · Выходы:{' '}
                  {telemetry?.networkData?.outputs.length ?? 0}
                </span>
              </div>
              <NeuralNetVisualizer data={telemetry?.networkData ?? null} />
            </div>
          )}

          {/* TAB 4: PROGRESS CHART & AGENTS TABLE */}
          {dockTab === 'chart' && (
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
              <div>
                <div className="mb-1 text-xs font-medium text-slate-300">
                  Максимальная дистанция по поколениям
                </div>
                <FitnessChart history={telemetry?.history ?? []} show30mTarget={isBiomechanics} />
              </div>
              <div className="max-h-[168px] overflow-y-auto rounded border border-slate-800 bg-slate-950/60">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-slate-900 text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="px-2.5 py-1.5 font-medium">Агент</th>
                      <th className="px-2.5 py-1.5 font-medium">Статус</th>
                      {isManualCar && <th className="px-2.5 py-1.5 font-medium text-right">КПП / RPM</th>}
                      {isDrone && <th className="px-2.5 py-1.5 font-medium text-right">Кольца</th>}
                      <th className="px-2.5 py-1.5 font-medium text-right">Дист.</th>
                      <th className="px-2.5 py-1.5 font-medium text-right">Балл</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono-tabular">
                    {(telemetry?.agents ?? [])
                      .slice()
                      .sort((a, b) => b.fitness - a.fitness)
                      .map((a) => (
                        <tr key={a.id} className={a.id === tracked?.id ? 'bg-amber-500/10' : ''}>
                          <td className="px-2.5 py-1 text-slate-200">#{a.id}</td>
                          <td className="px-2.5 py-1">
                            {a.alive ? (
                              <span className="text-emerald-400 font-sans">Активен</span>
                            ) : (
                              <span className="text-slate-500 font-sans truncate max-w-[140px] inline-block align-bottom">
                                {a.disqualifiedReason ?? 'Выбыл'}
                              </span>
                            )}
                          </td>
                          {isManualCar && (
                            <td className="px-2.5 py-1 text-right text-amber-300">
                              {formatGear(a.gear)} · {a.rpm ?? 900}
                            </td>
                          )}
                          {isDrone && (
                            <td className="px-2.5 py-1 text-right text-emerald-300">
                              {a.checkpointsPassed ?? 0}
                            </td>
                          )}
                          <td className="px-2.5 py-1 text-right text-slate-200">{a.maxDistance.toFixed(2)}м</td>
                          <td className="px-2.5 py-1 text-right text-sky-300">{a.fitness.toFixed(0)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 5: PER-USER DATABASE & CHECKPOINTS */}
          {dockTab === 'database' && (
            <div className="mt-3">
              {!user ? (
                <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-800 bg-slate-900/70 p-4 text-xs">
                  <div>
                    <div className="font-semibold text-slate-100">
                      Персональная облачная база поколений и весов нейросети
                    </div>
                    <p className="mt-1 text-slate-400">
                      Войдите через Google, чтобы автоматически сохранять номер поколения, рекорды по каждому режиму и именованные чекпоинты чемпионов.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleGoogleSignIn}
                    disabled={authBusy}
                    className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-sky-400 transition-colors cursor-pointer"
                  >
                    <LogIn className="h-4 w-4" />
                    <span>Войти через Google</span>
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  {/* Left: Save Checkpoint & Auto-saved Mode States */}
                  <div className="flex flex-col gap-2">
                    <form onSubmit={handleSaveCheckpoint} className="flex gap-1.5">
                      <input
                        type="text"
                        value={checkpointName}
                        onChange={(e) => setCheckpointName(e.target.value)}
                        placeholder={`Название чекпоинта (${MODE_NAMES[mode]})...`}
                        className="flex-1 rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:border-amber-500 focus:outline-none"
                      />
                      <button
                        type="submit"
                        disabled={savingCheckpoint}
                        className="inline-flex items-center gap-1 rounded-md bg-amber-500 px-3 py-1.5 font-semibold text-slate-950 hover:bg-amber-400 transition-colors whitespace-nowrap cursor-pointer disabled:opacity-50"
                      >
                        <BookmarkPlus className="h-3.5 w-3.5" />
                        <span>Сохранить</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleSaveNow}
                        title="Сохранить прогресс текущего режима"
                        className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-slate-200 hover:bg-slate-700 cursor-pointer"
                      >
                        <Cloud className="h-3.5 w-3.5 text-emerald-400" />
                      </button>
                    </form>

                    <div className="max-h-[120px] overflow-y-auto rounded border border-slate-800 bg-slate-950/60 divide-y divide-slate-800/60">
                      {savedModeStates.length === 0 ? (
                        <div className="p-3 text-slate-500">
                          Автосохранения по режимам появятся после 1-го завершённого поколения.
                        </div>
                      ) : (
                        savedModeStates.map((st) => (
                          <div key={st.mode} className="flex items-center justify-between px-2.5 py-1.5">
                            <div>
                              <span className="font-medium text-slate-200">{MODE_NAMES[st.mode]}</span>
                              <span className="ml-2 font-mono-tabular text-slate-400">
                                Пок. {st.generation} · Рекорд {st.bestDistanceEver.toFixed(1)}м
                              </span>
                            </div>
                            <button
                              type="button"
                              onClick={() => handleLoadModeState(st)}
                              className="text-sky-400 hover:text-sky-300 font-medium cursor-pointer"
                            >
                              Продолжить
                            </button>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Right: Saved Named Checkpoints */}
                  <div className="max-h-[162px] overflow-y-auto rounded border border-slate-800 bg-slate-950/60 divide-y divide-slate-800/60">
                    {checkpoints.length === 0 ? (
                      <div className="p-3 text-slate-500">
                        Сохранённых чекпоинтов пока нет. Нажмите «Сохранить», чтобы зафиксировать геном чемпиона.
                      </div>
                    ) : (
                      checkpoints.map((cp) => (
                        <div key={cp.id} className="flex items-center justify-between gap-2 px-2.5 py-1.5">
                          <div className="min-w-0">
                            <div className="truncate font-medium text-slate-100">{cp.name}</div>
                            <div className="font-mono-tabular text-[11px] text-slate-400">
                              {MODE_NAMES[cp.mode]} · Пок. {cp.generation} · {cp.bestDistance.toFixed(1)}м · {cp.activationFn}
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <button
                              type="button"
                              onClick={() => handleLoadCheckpoint(cp)}
                              className="rounded border border-sky-500/40 bg-sky-500/15 px-2 py-1 text-[11px] font-medium text-sky-300 hover:bg-sky-500/25 cursor-pointer"
                            >
                              В 3D
                            </button>
                            <button
                              type="button"
                              onClick={() => deleteUserCheckpoint(user.uid, cp.id)}
                              title="Удалить чекпоинт"
                              className="p-1 text-slate-500 hover:text-rose-400 cursor-pointer"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
