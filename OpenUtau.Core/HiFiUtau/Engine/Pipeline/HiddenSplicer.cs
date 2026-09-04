using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// Mel 域拼接 + OpenUtau vocoder oudep（PC-NSF-HiFiGAN 整图 ONNX：mel+f0 → wav）。
    /// 不使用 part1/part2 拆分图，也不加载 PyTorch ckpt。
    /// </summary>
    public sealed class HiddenSplicer : IDisposable {
        public int ModelHop { get; }
        public int SampleRate { get; }
        public int NumMels { get; }
        public int FrontPadFrames { get; } = 6;
        public int TailPadFrames { get; } = 4;

        readonly InferenceSession fused;

        public HiddenSplicer(InferenceSession fusedSession, int hopSize, int sampleRate, int numMels) {
            fused = fusedSession;
            ModelHop = hopSize > 0 ? hopSize : 512;
            SampleRate = sampleRate > 0 ? sampleRate : 44100;
            NumMels = numMels > 0 ? numMels : 128;
        }

        /// <summary>mel 域能量叠加后，用 vocoder oudep 整图合成。</summary>
        public float[] SpliceAndSynthesizeMel(List<PhoneInfo> phonemeList, double[] f0Np) {
            int n = phonemeList.Count;
            var last = phonemeList[^1];
            int maxMelLen = last.MelEnd + 1;
            int melDim = NumMels;
            var totalMelEnergy = new double[melDim * maxMelLen];

            for (int i = 0; i < n; i++) {
                var phoneme = phonemeList[i];
                double p0Y = phoneme.P0.Y, p4Y = phoneme.P4.Y;
                if (i == 0) p0Y = 100;
                if (i == n - 1) p4Y = 100;

                double preutter = phoneme.Preutter;
                double x0 = preutter + phoneme.P0.X * (SampleRate / 1000.0);
                double x1 = preutter + phoneme.P1.X * (SampleRate / 1000.0);
                double x2 = preutter + phoneme.P2.X * (SampleRate / 1000.0);
                double x3 = preutter + phoneme.P3.X * (SampleRate / 1000.0);
                double x4 = preutter + phoneme.P4.X * (SampleRate / 1000.0);

                double y0 = p0Y / 100, y1 = phoneme.P1.Y / 100, y2 = phoneme.P2.Y / 100;
                double y3 = phoneme.P3.Y / 100, y4 = p4Y / 100;

                var xs = new[] { x0, x1, x2, x3, x4 };
                var ys = new[] { y0, y1, y2, y3, y4 };
                var gain = new double[phoneme.HPoints.Length];
                for (int t = 0; t < gain.Length; t++) {
                    gain[t] = NpInterpScalar(phoneme.HPoints[t], xs, ys);
                }
                var mel = phoneme.Mel;
                int T = phoneme.MelFrames;
                var energyMel = new double[mel.Length];
                for (int k = 0; k < mel.Length; k++) energyMel[k] = Math.Exp(mel[k] * 2);
                for (int m = 0; m < melDim; m++) {
                    for (int t = 0; t < T; t++) energyMel[m * T + t] *= gain[t];
                }

                int start = phoneme.MelOffset;
                int clipStart = Math.Max(start, 0);
                int srcStart = clipStart - start;
                for (int m = 0; m < melDim; m++) {
                    for (int t = srcStart; t < T; t++) {
                        int dst = start + t;
                        if (dst < maxMelLen) totalMelEnergy[m * maxMelLen + dst] += energyMel[m * T + t];
                    }
                }
            }

            var totalMelLog = new double[melDim * maxMelLen];
            for (int i = 0; i < totalMelLog.Length; i++) {
                totalMelLog[i] = Math.Log(Math.Max(totalMelEnergy[i], 1e-12)) / 2;
            }

            var f0 = new double[maxMelLen];
            Array.Copy(f0Np, f0, Math.Min(f0Np.Length, maxMelLen));
            if (f0Np.Length < maxMelLen) {
                double edge = f0Np.Length > 0 ? f0Np[^1] : 440;
                for (int i = f0Np.Length; i < maxMelLen; i++) f0[i] = edge;
            }
            var f0Pad = new double[FrontPadFrames + maxMelLen + TailPadFrames];
            for (int t = 0; t < FrontPadFrames; t++) f0Pad[t] = f0[0];
            Array.Copy(f0, 0, f0Pad, FrontPadFrames, maxMelLen);
            for (int t = 0; t < TailPadFrames; t++) f0Pad[FrontPadFrames + maxMelLen + t] = f0[^1];

            var melPad = new double[melDim * (FrontPadFrames + maxMelLen + TailPadFrames)];
            int paddedLen = FrontPadFrames + maxMelLen + TailPadFrames;
            for (int m = 0; m < melDim; m++) {
                for (int t = 0; t < FrontPadFrames; t++) melPad[m * paddedLen + t] = totalMelLog[m * maxMelLen];
                Array.Copy(totalMelLog, m * maxMelLen, melPad, m * paddedLen + FrontPadFrames, maxMelLen);
                for (int t = 0; t < TailPadFrames; t++) {
                    melPad[m * paddedLen + FrontPadFrames + maxMelLen + t] = totalMelLog[m * maxMelLen + maxMelLen - 1];
                }
            }

            return FusedSynthesize(melPad, paddedLen, f0Pad);
        }

        /// <summary>
        /// vocoder oudep：mel [1, T, n_mels]，f0 [1, T]。
        /// HiFi 内部 mel 是 band-major (n_mels, T)，转置后再喂。
        /// </summary>
        float[] FusedSynthesize(double[] bandMajorMel, int frames, double[] f0) {
            var melNtc = new float[frames * NumMels];
            for (int t = 0; t < frames; t++) {
                for (int m = 0; m < NumMels; m++) {
                    melNtc[t * NumMels + m] = (float)bandMajorMel[m * frames + t];
                }
            }
            int f0Len = Math.Max(frames, f0.Length);
            var f0In = new float[f0Len];
            int n = Math.Min(f0.Length, f0Len);
            for (int i = 0; i < n; i++) f0In[i] = (float)f0[i];
            for (int i = n; i < f0Len; i++) f0In[i] = n > 0 ? f0In[n - 1] : 440f;

            var melTensor = new DenseTensor<float>(melNtc, new[] { 1, frames, NumMels });
            var f0Tensor = new DenseTensor<float>(f0In, new[] { 1, f0Len });
            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("mel", melTensor),
                NamedOnnxValue.CreateFromTensor("f0", f0Tensor),
            };
            using var results = fused.Run(inputs);
            var wav = results.First().AsTensor<float>();
            int total = (int)wav.Length;
            var samples = new float[total];
            for (int i = 0; i < total; i++) samples[i] = wav.GetValue(i);
            return samples;
        }

        static double NpInterpScalar(double x, double[] xp, double[] fp) {
            if (x <= xp[0]) return fp[0];
            if (x >= xp[^1]) return fp[^1];
            int j = Array.BinarySearch(xp, x);
            if (j >= 0) return fp[j];
            j = ~j - 1;
            double t = (x - xp[j]) / (xp[j + 1] - xp[j]);
            return fp[j] + t * (fp[j + 1] - fp[j]);
        }

        public void Dispose() {
            fused.Dispose();
        }
    }
}
