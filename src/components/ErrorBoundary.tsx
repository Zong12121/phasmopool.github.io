import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface State {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  State
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  public static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      errorMessage: error?.message || 'Непредвиденная ошибка приложения',
    };
  }

  public render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex h-screen w-screen items-center justify-center bg-[#080b10] p-6 text-slate-100">
          <div className="max-w-lg rounded-xl border border-rose-500/40 bg-slate-900/95 p-6 shadow-2xl">
            <div className="flex items-center gap-3 text-rose-400">
              <AlertTriangle className="h-6 w-6 shrink-0" />
              <h2 className="font-display text-lg font-semibold">
                Ошибка лаборатории нейроэволюции
              </h2>
            </div>
            <p className="mt-3 rounded-lg border border-slate-800 bg-slate-950/80 p-3 font-mono text-xs text-slate-300 break-words">
              {this.state.errorMessage}
            </p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="mt-4 inline-flex items-center gap-2 rounded-lg bg-amber-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-amber-400 transition-colors cursor-pointer"
            >
              <RotateCcw className="h-4 w-4" />
              <span>Перезагрузить лабораторию</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
