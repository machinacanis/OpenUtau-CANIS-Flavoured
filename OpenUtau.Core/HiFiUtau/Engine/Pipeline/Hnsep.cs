using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core.HiFiUtau.Engine.Dsp;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// tools/hnsep_onnx.py 的移植：谐波/噪声分离。
    /// pt2 新模型: 手写 STFT（对称 hann、reflect 填充）→ mask 推理 → 手写 ISTFT。
    /// 旧模型: waveform → harmonic + noise。
    /// </summary>
    public sealed class Hnsep : IDisposable {
        private const int Nfft = 2048;
        private const int HopLength = 512;
        private const int SegLength = 32 * HopLength;

        private readonly InferenceSession session;
        private readonly bool isPt2;

        public Hnsep(string modelPath, Func<string, InferenceSession> sessionFactory) {
            isPt2 = modelPath.Contains("pt2");
            session = sessionFactory(modelPath);
        }

        public (double[] harmonic, double[] noise) Separate(float[] waveform) {
            return isPt2 ? SeparatePt2(waveform) : SeparateLegacy(waveform);
        }

        private (double[] harmonic, double[] noise) SeparateLegacy(float[] waveform) {
            var input = new float[waveform.Length];
            Array.Copy(waveform, input, waveform.Length);
            var tensor = new DenseTensor<float>(input, new[] { 1, waveform.Length });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("waveform", tensor) };
            using var results = session.Run(inputs);
            var h = results.First(r => r.Name == "harmonic").AsTensor<float>();
            var n = results.First(r => r.Name == "noise").AsTensor<float>();
            var harmonic = new double[h.Length];
            var noise = new double[n.Length];
            for (int i = 0; i < h.Length; i++) harmonic[i] = h.GetValue(i);
            for (int i = 0; i < n.Length; i++) noise[i] = n.GetValue(i);
            return (harmonic, noise);
        }

        private (double[] harmonic, double[] noise) SeparatePt2(float[] waveform) {
            int nSamples = waveform.Length;
            var window = Stft.HannSymmetric(Nfft);  // np.hanning

            int t1 = nSamples + HopLength;
            int tPad = SegLength * ((t1 - 1) / SegLength + 1) - t1;
            int nlPad = tPad / 2 / HopLength;
            int tlPad = nlPad * HopLength;
            var padded = new float[tlPad + nSamples + (tPad - tlPad)];
            Array.Copy(waveform, 0, padded, tlPad, nSamples);

            // reflect 填充 n_fft/2（Python 显式 reflect）
            int padLen = Nfft / 2;
            var buf = Stft.PadReflect(padded, padLen, padLen);
            int nFrames = (buf.Length - Nfft) / HopLength + 1;
            int bins = Nfft / 2 + 1;

            // STFT（bin-major [1025, T]）
            var specRe = new float[bins * nFrames];
            var specIm = new float[bins * nFrames];
            var frame = new float[Nfft];
            var frameIm = new float[Nfft];
            var reOut = new float[Nfft];
            var imOut = new float[Nfft];
            var fft = new NWaves.Transforms.Fft(Nfft);
            for (int t = 0; t < nFrames; t++) {
                int off = t * HopLength;
                for (int i = 0; i < Nfft; i++) {
                    frame[i] = buf[off + i] * window[i];
                    frameIm[i] = 0f;
                }
                fft.Direct(frame, frameIm, reOut, imOut);
                for (int b = 0; b < bins; b++) {
                    specRe[b * nFrames + t] = reOut[b];
                    specIm[b * nFrames + t] = imOut[b];
                }
            }

            // mask 推理（模型 Concat 节点要求 spec_real/spec_imag 同时输入）
            var maskRe = new float[bins * nFrames];
            var maskIm = new float[bins * nFrames];
            {
                var reTensor = new DenseTensor<float>(specRe, new[] { 1, 1, bins, nFrames });
                var imTensor = new DenseTensor<float>(specIm, new[] { 1, 1, bins, nFrames });
                var inputs = new List<NamedOnnxValue> {
                    NamedOnnxValue.CreateFromTensor("spec_real", reTensor),
                    NamedOnnxValue.CreateFromTensor("spec_imag", imTensor),
                };
                using var results = session.Run(inputs);
                var maskReTensor = results.First(r => r.Name == "mask_real").AsTensor<float>();
                var maskImTensor = results.First(r => r.Name == "mask_imag").AsTensor<float>();
                for (int i = 0; i < maskRe.Length; i++) {
                    maskRe[i] = maskReTensor.GetValue(i);
                    maskIm[i] = maskImTensor.GetValue(i);
                }
            }

            // spec * (mask_r + j·mask_i) → ISTFT 重叠相加
            var harmonic = new double[buf.Length];
            var norm = new double[buf.Length];
            for (int t = 0; t < nFrames; t++) {
                // 构建共轭对称全谱
                var fullRe = new float[Nfft];
                var fullIm = new float[Nfft];
                for (int b = 0; b < bins; b++) {
                    double r = specRe[b * nFrames + t], im = specIm[b * nFrames + t];
                    double mr = maskRe[b * nFrames + t], mi = maskIm[b * nFrames + t];
                    fullRe[b] = (float)(r * mr - im * mi);
                    fullIm[b] = (float)(r * mi + im * mr);
                }
                for (int b = 1; b < Nfft - bins + 1; b++) {
                    fullRe[Nfft - b] = fullRe[b];
                    fullIm[Nfft - b] = -fullIm[b];
                }
                var timeRe = new float[Nfft];
                var timeIm = new float[Nfft];
                fft.InverseNorm(fullRe, fullIm, timeRe, timeIm);
                int off = t * HopLength;
                for (int i = 0; i < Nfft; i++) {
                    harmonic[off + i] += (double)timeRe[i] * window[i];
                    norm[off + i] += (double)window[i] * window[i];
                }
            }
            for (int i = 0; i < harmonic.Length; i++) {
                harmonic[i] /= Math.Max(norm[i], 1e-10);
            }

            // 裁剪填充
            int start = padLen + tlPad;
            var hOut = new double[nSamples];
            Array.Copy(harmonic, start, hOut, 0, nSamples);
            var nOut = new double[nSamples];
            for (int i = 0; i < nSamples; i++) nOut[i] = waveform[i] - hOut[i];
            return (hOut, nOut);
        }

        public void Dispose() => session?.Dispose();
    }
}
