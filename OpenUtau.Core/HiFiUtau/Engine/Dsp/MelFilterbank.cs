using System;

namespace OpenUtau.Core.HiFiUtau.Engine.Dsp {
    /// <summary>
    /// librosa.filters.mel(htk=false, norm='slaney') 的精确移植。
    /// Slaney mel 刻度（200/3 线性段 + log(6.4)/27 对数段）与面积归一化，
    /// 基矩阵先以 double 计算再 round 到 float32——与 librosa 的 float32 基矩阵
    /// 在 float64 域做 dot 的实际行为完全一致。
    /// </summary>
    public static class MelFilterbank {
        public const double SlaneyLinearStep = 200.0 / 3.0;   // Hz per mel below 1kHz
        public static readonly double SlaneyLogStep = Math.Log(6.4) / 27.0;
        public const double SlaneyBreakHz = 1000.0;

        public static double HzToMel(double hz) {
            return hz < SlaneyBreakHz
                ? hz / SlaneyLinearStep
                : 15.0 + Math.Log(hz / SlaneyBreakHz) / SlaneyLogStep;
        }

        public static double MelToHz(double mel) {
            return mel < 15.0
                ? mel * SlaneyLinearStep
                : SlaneyBreakHz * Math.Exp(SlaneyLogStep * (mel - 15.0));
        }

        public static double[] MelFrequencies(int n, double fmin, double fmax) {
            double melMin = HzToMel(fmin), melMax = HzToMel(fmax);
            var result = new double[n];
            for (int i = 0; i < n; i++) {
                double mel = n == 1 ? melMin : melMin + (melMax - melMin) * i / (n - 1);
                result[i] = MelToHz(mel);
            }
            return result;
        }

        /// <summary>返回平铺 (nMels, bins) 的基矩阵，bins = nFft/2+1，band-major。</summary>
        public static double[] Basis(int sr, int nFft, int nMels, double fmin, double fmax) {
            int bins = nFft / 2 + 1;
            var fftFreqs = new double[bins];
            for (int i = 0; i < bins; i++) {
                fftFreqs[i] = sr / 2.0 * i / (bins - 1);
            }
            var melF = MelFrequencies(nMels + 2, fmin, fmax);
            var w = new double[nMels * bins];
            for (int m = 0; m < nMels; m++) {
                double left = melF[m], center = melF[m + 1], right = melF[m + 2];
                double enorm = 2.0 / (melF[m + 2] - melF[m]);
                for (int b = 0; b < bins; b++) {
                    double lower = center > left ? (fftFreqs[b] - left) / (center - left) : 0.0;
                    double upper = right > center ? (right - fftFreqs[b]) / (right - center) : 0.0;
                    double v = Math.Max(0.0, Math.Min(lower, upper));
                    w[m * bins + b] = v * enorm;
                }
            }
            // round 到 float32 再回 double，与 librosa float32 基矩阵一致
            for (int i = 0; i < w.Length; i++) {
                w[i] = (double)(float)w[i];
            }
            return w;
        }
    }
}
