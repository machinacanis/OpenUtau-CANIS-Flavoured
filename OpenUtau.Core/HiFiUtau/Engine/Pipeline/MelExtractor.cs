using System;
using System.Threading.Tasks;
using OpenUtau.Core.HiFiUtau.Engine.Dsp;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// util/wav2mel.py 的 PitchAndTimeAdjustableMelSpectrogram 移植：
    /// 支持自定义帧中心位置（用于时间拉伸）的 mel 提取。
    /// STFT 用对称 hann（np.hanning）+ reflect 填充（Python 显式指定，与 librosa 版本无关）。
    /// 输出 (128, T) band-major。
    /// </summary>
    public sealed class MelExtractor {
        public const int Nfft = 2048;
        public const int WinLength = 2048;
        public const int NMels = 128;
        public const double Fmin = 40, Fmax = 16000;
        public const int SampleRate = 44100;

        private readonly float[] melBasisF32; // float32 mel 基（librosa 同为 float32），SIMD 点积用
        private readonly float[] hannSym;     // np.hanning

        public MelExtractor() {
            melBasisF32 = Simd.ToFloat32(MelFilterbank.Basis(SampleRate, Nfft, NMels, Fmin, Fmax));
            hannSym = Stft.HannSymmetric(WinLength);
        }

        /// <summary>
        /// wav2mel __call__（key_shift=0 路径，引擎中仅使用该路径）。
        /// centers: 帧中心采样位置（float 截断取整）。
        /// </summary>
        public double[] Extract(float[] y, double[] centers) {
            int T = centers.Length;
            int bins = Nfft / 2 + 1;
            var padded = Stft.PadReflect(y, Nfft / 2, Nfft / 2);
            int padLen = padded.Length;

            // 幅度谱存为帧主序 [t*bins+b]，mel 投影两侧均连续（SIMD 点积 + 按频带并行）
            var magT = new float[bins * T];
            var frame = new float[Nfft];
            var frameIm = new float[Nfft];
            var reOut = new float[Nfft];
            var imOut = new float[Nfft];
            var fft = new NWaves.Transforms.Fft(Nfft);

            for (int t = 0; t < T; t++) {
                int start = (int)centers[t];  // np: indices.astype(int) 截断
                for (int i = 0; i < Nfft; i++) {
                    int j = start + i;
                    // Python: np.clip(indices, 0, len(padded)-1) 后取帧
                    j = Math.Clamp(j, 0, padLen - 1);
                    frame[i] = padded[j] * hannSym[i];
                    frameIm[i] = 0f;
                }
                fft.Direct(frame, frameIm, reOut, imOut);
                int dst = t * bins;
                for (int b = 0; b < bins; b++) {
                    float r = reOut[b], m = imOut[b];
                    magT[dst + b] = MathF.Sqrt(r * r + m * m);
                }
            }

            // mel = basis @ |spec|（basis 为 float32-round 后的 double，与 librosa 一致）
            // 小输入并行开销大于收益，仅在帧数较多时按频带并行
            var mel = new double[NMels * T];
            if (T >= 512) {
                Parallel.For(0, NMels, m => {
                    int off = m * bins;
                    int outOff = m * T;
                    for (int t = 0; t < T; t++) {
                        mel[outOff + t] = Simd.Dot(melBasisF32, off, magT, t * bins, bins);
                    }
                });
            } else {
                for (int m = 0; m < NMels; m++) {
                    int off = m * bins;
                    int outOff = m * T;
                    for (int t = 0; t < T; t++) {
                        mel[outOff + t] = Simd.Dot(melBasisF32, off, magT, t * bins, bins);
                    }
                }
            }
            return mel;
        }
    }
}
