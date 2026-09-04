using System;
using NWaves.Transforms;

namespace OpenUtau.Core.HiFiUtau.Engine.Dsp {
    /// <summary>
    /// librosa 语义的 STFT/ISTFT 薄胶水层。
    ///
    /// 与 librosa.stft/istft(center=True, window='hann') 逐点对齐：
    ///   - 周期性 hann 窗（scipy get_window fftbins=True）
    ///   - librosa 0.11 起默认 pad_mode='constant'（零填充）；0.10 为 'reflect'。
    ///     引擎管线中的 mel/tension/lowcut/warmth 均未显式传 pad_mode，
    ///     随 librosa 版本变化——本实现提供两种模式，默认 constant（与当前
    ///     已发布 PyInstaller 引擎一致）；hnsep/wav2mel 在 Python 中显式用 reflect。
    ///   - ISTFT 为 window² 重叠相加归一化，center 裁剪后输出 hop*(T-1) 个采样
    /// FFT 内核使用 NWaves.Fft（float32 内部精度，与 numpy float64 相差 ~1e-6 量级，
    /// 经金标准测试验证对合成结果无可闻影响）。
    /// 频谱布局与 numpy 一致：bin-major 平铺，索引 [bin * frames + t]，bins = nFft/2+1。
    /// </summary>
    public static class Stft {
        public enum PadMode { Constant, Reflect }

        /// <summary>周期性 hann（scipy get_window('hann', N, fftbins=True)）。</summary>
        public static float[] HannPeriodic(int n) {
            var w = new float[n];
            for (int i = 0; i < n; i++) {
                w[i] = (float)(0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / n)));
            }
            return w;
        }

        /// <summary>对称 hann（np.hanning / NWaves Window.Hann）。</summary>
        public static float[] HannSymmetric(int n) {
            return NWaves.Windows.Window.Hann(n);
        }

        /// <summary>np.pad(x, (left, right), mode='reflect')。</summary>
        public static float[] PadReflect(float[] x, int left, int right) {
            int n = x.Length;
            var outArr = new float[left + n + right];
            for (int i = 0; i < left; i++) {
                int src = ReflectIndex(i, left, n);
                outArr[i] = x[src];
            }
            Array.Copy(x, 0, outArr, left, n);
            for (int i = 0; i < right; i++) {
                int src = ReflectIndex(left + n + i, left, n);
                outArr[left + n + i] = x[src];
            }
            return outArr;
        }

        private static int ReflectIndex(int idx, int left, int n) {
            // np.pad reflect: 关于边界元素镜像，不含边界本身
            int span = 2 * (n - 1);
            int v = (idx - left) % span;
            if (v < 0) v += span;
            return v < n - 1 ? v : span - v;
        }

        /// <summary>np.pad(x, (left, right), mode='constant')（零填充）。</summary>
        public static float[] PadConstant(float[] x, int left, int right) {
            var outArr = new float[left + x.Length + right];
            Array.Copy(x, 0, outArr, left, x.Length);
            return outArr;
        }

        public static (float[] re, float[] im, int bins, int frames) Forward(
                float[] x, int nFft, int hop, float[] win, bool center,
                PadMode padMode = PadMode.Constant) {
            int bins = nFft / 2 + 1;
            float[] buf = x;
            if (center) {
                int pad = nFft / 2;
                buf = padMode == PadMode.Reflect ? PadReflect(x, pad, pad) : PadConstant(x, pad, pad);
            }
            int frames = buf.Length < nFft ? 1 : (buf.Length - nFft) / hop + 1;
            // librosa: 短于 n_fft 的信号也至少产生 1 帧（窗口右侧补零语义），
            // 这里与 librosa 保持一致：不足部分补零凑满一帧。
            var re = new float[bins * frames];
            var im = new float[bins * frames];

            int winLen = win.Length;
            var winPadded = new float[nFft];
            int padLeft = (nFft - winLen) / 2;
            Array.Copy(win, 0, winPadded, padLeft, winLen);

            var fft = new Fft(nFft);
            var frameRe = new float[nFft];
            var frameIm = new float[nFft];
            var outRe = new float[nFft];
            var outIm = new float[nFft];

            for (int t = 0; t < frames; t++) {
                int off = t * hop;
                for (int i = 0; i < nFft; i++) {
                    int j = off + i;
                    frameRe[i] = j < buf.Length ? buf[j] * winPadded[i] : 0f;
                    frameIm[i] = 0f;
                }
                fft.Direct(frameRe, frameIm, outRe, outIm);
                for (int b = 0; b < bins; b++) {
                    re[b * frames + t] = outRe[b];
                    im[b * frames + t] = outIm[b];
                }
            }
            return (re, im, bins, frames);
        }

        /// <summary>
        /// librosa.istft(center=True, length=None) 等价：输出长度 = hop*(frames-1)。
        /// re/im 为 bin-major 平铺的半谱。
        /// </summary>
        public static float[] Inverse(float[] re, float[] im, int nFft, int hop, float[] win, bool center) {
            int bins = nFft / 2 + 1;
            int frames = re.Length / bins;

            int winLen = win.Length;
            var winPadded = new float[nFft];
            int padLeft = (nFft - winLen) / 2;
            Array.Copy(win, 0, winPadded, padLeft, winLen);

            int nOut = nFft + hop * (frames - 1);
            var acc = new double[nOut];
            var norm = new double[nOut];

            var fft = new Fft(nFft);
            var fullRe = new float[nFft];
            var fullIm = new float[nFft];
            var timeRe = new float[nFft];
            var timeIm = new float[nFft];

            for (int t = 0; t < frames; t++) {
                // 半谱 → 共轭对称全谱（librosa: concatenate([S, conj(S[-2:0:-1])])）
                for (int b = 0; b < bins; b++) {
                    fullRe[b] = re[b * frames + t];
                    fullIm[b] = im[b * frames + t];
                }
                for (int b = 1; b < nFft - bins + 1; b++) {
                    fullRe[nFft - b] = fullRe[b];
                    fullIm[nFft - b] = -fullIm[b];
                }
                fft.InverseNorm(fullRe, fullIm, timeRe, timeIm);
                int off = t * hop;
                for (int i = 0; i < nFft; i++) {
                    acc[off + i] += (double)timeRe[i] * winPadded[i];
                    norm[off + i] += (double)winPadded[i] * winPadded[i];
                }
            }

            var y = new double[nOut];
            for (int i = 0; i < nOut; i++) {
                y[i] = norm[i] > 1e-12 ? acc[i] / norm[i] : 0.0;
            }

            int start = center ? nFft / 2 : 0;
            int len = center ? nOut - nFft : nOut;
            var result = new float[len];
            for (int i = 0; i < len; i++) {
                result[i] = (float)y[start + i];
            }
            return result;
        }
    }
}
