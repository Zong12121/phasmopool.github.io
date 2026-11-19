import React from 'react';
import type { GenerationHistoryPoint } from '../simulation/types';

interface Props {
  history: GenerationHistoryPoint[];
  show30mTarget?: boolean;
}

export const FitnessChart: React.FC<Props> = ({ history, show30mTarget = true }) => {
  const width = 420;
  const height = 148;
  const padL = 36;
  const padR = 14;
  const padT = 14;
  const padB = 22;

  if (history.length === 0) {
    return (
      <div className="flex h-[148px] items-center justify-center rounded border border-slate-800/80 bg-slate-950/60 text-xs text-slate-400">
        График появится после завершения 1-го поколения
      </div>
    );
  }

  const maxDist = Math.max(show30mTarget ? 32 : 5, ...history.map((h) => h.bestDistance));
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const getX = (idx: number) =>
    history.length === 1 ? padL + plotW / 2 : padL + (idx / (history.length - 1)) * plotW;
  const getYDist = (d: number) => padT + plotH - (Math.max(0, d) / maxDist) * plotH;

  const distPoints = history.map((h, idx) => `${getX(idx).toFixed(1)},${getYDist(h.bestDistance).toFixed(1)}`).join(' ');
  const y30 = getYDist(30);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-[148px] select-none">
      {/* Grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const y = padT + plotH * (1 - t);
        const label = (maxDist * t).toFixed(0);
        return (
          <g key={t}>
            <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="rgba(148,163,184,0.12)" strokeWidth={1} />
            <text x={padL - 6} y={y + 3} textAnchor="end" className="fill-slate-500 text-[9px] font-mono">
              {label}м
            </text>
          </g>
        );
      })}

      {/* 30m Challenge Threshold Line */}
      {show30mTarget && y30 >= padT && y30 <= padT + plotH && (
        <g>
          <line
            x1={padL}
            y1={y30}
            x2={width - padR}
            y2={y30}
            stroke="rgba(245, 158, 11, 0.6)"
            strokeDasharray="4 3"
            strokeWidth={1.2}
          />
          <text x={width - padR - 2} y={y30 - 4} textAnchor="end" className="fill-amber-400 text-[8.5px] font-mono">
            Порог 30 м
          </text>
        </g>
      )}

      {/* Distance Polyline */}
      {history.length > 1 && (
        <polyline
          fill="none"
          stroke="#38bdf8"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
          points={distPoints}
        />
      )}

      {history.map((h, idx) => (
        <circle
          key={h.generation}
          cx={getX(idx)}
          cy={getYDist(h.bestDistance)}
          r={history.length > 20 ? 2 : 3}
          fill="#38bdf8"
        />
      ))}

      <text x={padL} y={height - 5} className="fill-slate-500 text-[9px] font-mono">
        Пок. {history[0].generation}
      </text>
      <text x={width - padR} y={height - 5} textAnchor="end" className="fill-slate-500 text-[9px] font-mono">
        Пок. {history[history.length - 1].generation}
      </text>
    </svg>
  );
};
