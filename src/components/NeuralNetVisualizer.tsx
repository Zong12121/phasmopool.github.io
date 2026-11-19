import React from 'react';
import type { NetworkInspectorData } from '../simulation/types';

interface Props {
  data: NetworkInspectorData | null;
}

export const NeuralNetVisualizer: React.FC<Props> = ({ data }) => {
  if (!data) {
    return (
      <div className="flex h-48 items-center justify-center text-xs text-slate-400">
        Ожидание телеметрии нейросети...
      </div>
    );
  }

  const { inputs, hidden, outputs, inputLabels, outputLabels, weightsHO, activationFn } = data;
  const width = 620;
  const height = Math.max(250, inputs.length * 15 + 24);

  const xIn = 155;
  const xHid = 335;
  const xOut = 485;

  const getY = (idx: number, count: number) => {
    const pad = 18;
    const avail = height - pad * 2;
    if (count === 1) return height / 2;
    return pad + (idx / (count - 1)) * avail;
  };

  const actColor = (val: number) => {
    const clamped = Math.max(-1, Math.min(1, val));
    if (clamped >= 0) {
      const alpha = 0.25 + clamped * 0.75;
      return `rgba(245, 158, 11, ${alpha.toFixed(2)})`; // Amber positive
    }
    const alpha = 0.25 + Math.abs(clamped) * 0.75;
    return `rgba(56, 189, 248, ${alpha.toFixed(2)})`; // Cyan negative
  };

  // Sample strongest hidden->output edges for clean legibility
  const hoEdges: Array<{ h: number; o: number; w: number }> = [];
  for (let o = 0; o < outputs.length; o++) {
    for (let h = 0; h < hidden.length; h++) {
      const w = weightsHO[o * hidden.length + h] ?? 0;
      if (Math.abs(w) > 0.22) {
        hoEdges.push({ h, o, w });
      }
    }
  }

  return (
    <div className="w-full overflow-hidden">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto max-h-[280px] select-none"
        role="img"
        aria-label="Топология и активации нейросети"
      >
        {activationFn && (
          <text
            x={width - 8}
            y={14}
            textAnchor="end"
            className="fill-slate-500 text-[8.5px] font-mono"
          >
            f(x): {activationFn}
          </text>
        )}
        {/* Subtle Input -> Hidden connections */}
        {inputs.map((_, i) =>
          hidden.map((__, h) => {
            if ((i + h) % 3 !== 0) return null;
            return (
              <line
                key={`ih-${i}-${h}`}
                x1={xIn}
                y1={getY(i, inputs.length)}
                x2={xHid}
                y2={getY(h, hidden.length)}
                stroke="rgba(148, 163, 184, 0.09)"
                strokeWidth={0.8}
              />
            );
          }),
        )}

        {/* Active Hidden -> Output weighted connections */}
        {hoEdges.map(({ h, o, w }) => {
          const signal = (hidden[h] ?? 0) * w;
          const pos = signal >= 0;
          const opacity = Math.min(0.85, 0.14 + Math.abs(signal) * 0.55);
          return (
            <line
              key={`ho-${h}-${o}`}
              x1={xHid}
              y1={getY(h, hidden.length)}
              x2={xOut}
              y2={getY(o, outputs.length)}
              stroke={pos ? `rgba(245, 158, 11, ${opacity})` : `rgba(56, 189, 248, ${opacity})`}
              strokeWidth={Math.min(2.4, 0.7 + Math.abs(w) * 0.8)}
            />
          );
        })}

        {/* Input Neurons */}
        {inputs.map((val, i) => {
          const y = getY(i, inputs.length);
          return (
            <g key={`in-${i}`}>
              <text
                x={xIn - 10}
                y={y + 3}
                textAnchor="end"
                className="fill-slate-400 text-[8.5px] font-medium"
              >
                {inputLabels[i] ?? `S${i + 1}`}
              </text>
              <circle
                cx={xIn}
                cy={y}
                r={4.2}
                fill={actColor(val)}
                stroke="rgba(226, 232, 240, 0.35)"
                strokeWidth={0.8}
              />
            </g>
          );
        })}

        {/* Hidden Neurons */}
        {hidden.map((val, h) => {
          const y = getY(h, hidden.length);
          return (
            <circle
              key={`hid-${h}`}
              cx={xHid}
              cy={y}
              r={4.8}
              fill={actColor(val)}
              stroke="rgba(226, 232, 240, 0.35)"
              strokeWidth={0.8}
            />
          );
        })}

        {/* Output Motor Neurons */}
        {outputs.map((val, o) => {
          const y = getY(o, outputs.length);
          return (
            <g key={`out-${o}`}>
              <circle
                cx={xOut}
                cy={y}
                r={6.2}
                fill={actColor(val)}
                stroke="rgba(248, 250, 252, 0.55)"
                strokeWidth={1.1}
              />
              <text
                x={xOut + 12}
                y={y + 3}
                textAnchor="start"
                className="fill-slate-200 text-[9.5px] font-medium"
              >
                {outputLabels[o] ?? `M${o + 1}`}{' '}
                <tspan className="fill-amber-300 font-mono">{val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}</tspan>
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};
