import type { ActivationFn, GenomeData } from '../simulation/types';

function activateHidden(x: number, fn: ActivationFn): number {
  if (fn === 'swish') {
    const cx = x > 12 ? 12 : x < -12 ? -12 : x;
    const sig = 1 / (1 + Math.exp(-cx));
    const v = cx * sig;
    return v > 3 ? 3 : v;
  }
  if (fn === 'leaky_relu') {
    const v = x >= 0 ? x : 0.12 * x;
    return v > 3.5 ? 3.5 : v < -1.5 ? -1.5 : v;
  }
  return Math.tanh(x);
}

export class NeuralNetwork {
  public readonly inputSize: number;
  public readonly hiddenSize: number;
  public readonly outputSize: number;
  public activationFn: ActivationFn;

  public wIH: Float32Array;
  public bH: Float32Array;
  public wHO: Float32Array;
  public bO: Float32Array;

  // Pre-allocated activation buffers to avoid GC in physics loops
  public readonly lastInputs: Float32Array;
  public readonly lastHidden: Float32Array;
  public readonly lastOutputs: Float32Array;

  constructor(
    inputSize: number,
    hiddenSize: number,
    outputSize: number,
    activationFn: ActivationFn = 'tanh',
  ) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.activationFn = activationFn;

    this.wIH = new Float32Array(inputSize * hiddenSize);
    this.bH = new Float32Array(hiddenSize);
    this.wHO = new Float32Array(hiddenSize * outputSize);
    this.bO = new Float32Array(outputSize);

    this.lastInputs = new Float32Array(inputSize);
    this.lastHidden = new Float32Array(hiddenSize);
    this.lastOutputs = new Float32Array(outputSize);

    this.randomize();
  }

  public randomize(scale = 0.92): void {
    const scaleIH = Math.sqrt(2 / (this.inputSize + this.hiddenSize)) * scale;
    const scaleHO = Math.sqrt(2 / (this.hiddenSize + this.outputSize)) * scale;

    for (let i = 0; i < this.wIH.length; i++) {
      this.wIH[i] = gaussianRandom() * scaleIH;
    }
    for (let i = 0; i < this.bH.length; i++) {
      this.bH[i] = gaussianRandom() * 0.18;
    }
    for (let i = 0; i < this.wHO.length; i++) {
      this.wHO[i] = gaussianRandom() * scaleHO;
    }
    for (let i = 0; i < this.bO.length; i++) {
      this.bO[i] = gaussianRandom() * 0.15;
    }
  }

  public forward(inputs: ArrayLike<number>): Float32Array {
    const { inputSize, hiddenSize, outputSize, wIH, bH, wHO, bO, activationFn } = this;

    for (let i = 0; i < inputSize; i++) {
      this.lastInputs[i] = inputs[i];
    }

    for (let h = 0; h < hiddenSize; h++) {
      let sum = bH[h];
      const offset = h * inputSize;
      for (let i = 0; i < inputSize; i++) {
        sum += inputs[i] * wIH[offset + i];
      }
      this.lastHidden[h] = activateHidden(sum, activationFn);
    }

    for (let o = 0; o < outputSize; o++) {
      let sum = bO[o];
      const offset = o * hiddenSize;
      for (let h = 0; h < hiddenSize; h++) {
        sum += this.lastHidden[h] * wHO[offset + h];
      }
      this.lastOutputs[o] = Math.tanh(sum);
    }

    return this.lastOutputs;
  }

  public clone(): NeuralNetwork {
    const copy = new NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize, this.activationFn);
    copy.wIH.set(this.wIH);
    copy.bH.set(this.bH);
    copy.wHO.set(this.wHO);
    copy.bO.set(this.bO);
    return copy;
  }

  public mutate(rate: number, strength: number): void {
    const mutateArray = (arr: Float32Array) => {
      for (let i = 0; i < arr.length; i++) {
        if (Math.random() < rate) {
          const r = Math.random();
          if (r < 0.06) {
            arr[i] = gaussianRandom() * 0.75;
          } else if (r < 0.12) {
            arr[i] *= -0.85; // Occasional sign flip escapes local optima
          } else {
            arr[i] += gaussianRandom() * strength;
          }
          if (arr[i] > 5.0) arr[i] = 5.0;
          if (arr[i] < -5.0) arr[i] = -5.0;
        }
      }
    };
    mutateArray(this.wIH);
    mutateArray(this.bH);
    mutateArray(this.wHO);
    mutateArray(this.bO);
  }

  public serialize(): GenomeData {
    const round = (arr: Float32Array) => Array.from(arr, (v) => Math.round(v * 10000) / 10000);
    return {
      inputSize: this.inputSize,
      hiddenSize: this.hiddenSize,
      outputSize: this.outputSize,
      activationFn: this.activationFn,
      wIH: round(this.wIH),
      bH: round(this.bH),
      wHO: round(this.wHO),
      bO: round(this.bO),
    };
  }

  public static deserialize(data: GenomeData): NeuralNetwork {
    const nn = new NeuralNetwork(
      data.inputSize,
      data.hiddenSize,
      data.outputSize,
      data.activationFn ?? 'tanh',
    );
    nn.wIH.set(data.wIH.slice(0, nn.wIH.length));
    nn.bH.set(data.bH.slice(0, nn.bH.length));
    nn.wHO.set(data.wHO.slice(0, nn.wHO.length));
    nn.bO.set(data.bO.slice(0, nn.bO.length));
    return nn;
  }

  public static crossover(a: NeuralNetwork, b: NeuralNetwork): NeuralNetwork {
    const child = new NeuralNetwork(a.inputSize, a.hiddenSize, a.outputSize, a.activationFn);
    const blend = Math.random() < 0.45;
    const cross = (dst: Float32Array, srcA: Float32Array, srcB: Float32Array) => {
      if (blend) {
        // BLX-style smooth weight interpolation preserves coupled motor motifs
        for (let i = 0; i < dst.length; i++) {
          const alpha = 0.25 + Math.random() * 0.5;
          dst[i] = srcA[i] * alpha + srcB[i] * (1 - alpha);
        }
      } else {
        for (let i = 0; i < dst.length; i++) {
          dst[i] = Math.random() < 0.5 ? srcA[i] : srcB[i];
        }
      }
    };
    cross(child.wIH, a.wIH, b.wIH);
    cross(child.bH, a.bH, b.bH);
    cross(child.wHO, a.wHO, b.wHO);
    cross(child.bO, a.bO, b.bO);
    return child;
  }
}

export function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
