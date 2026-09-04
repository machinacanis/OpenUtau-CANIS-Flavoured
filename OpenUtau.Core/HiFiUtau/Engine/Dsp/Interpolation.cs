using System;

namespace OpenUtau.Core.HiFiUtau.Engine.Dsp {
    /// <summary>
    /// numpy/scipy 语义的一维插值工具：
    ///   - NpInterp: np.interp（端点截断）
    ///   - LinearExtrap: interp1d(kind='linear', fill_value='extrapolate')
    ///   - InterpToLen / ResampleArray: 引擎内帧级参数重采样
    /// </summary>
    public static class Interpolation {
        /// <summary>np.interp：xp 升序，x 越界取端点值。</summary>
        public static double[] NpInterp(double[] xs, double[] xp, double[] fp) {
            var result = new double[xs.Length];
            int j = 0;
            for (int i = 0; i < xs.Length; i++) {
                double x = xs[i];
                if (x <= xp[0]) { result[i] = fp[0]; continue; }
                if (x >= xp[xp.Length - 1]) { result[i] = fp[fp.Length - 1]; continue; }
                while (j < xp.Length - 2 && xp[j + 1] < x) j++;
                while (j > 0 && xp[j] > x) j--;
                double t = (x - xp[j]) / (xp[j + 1] - xp[j]);
                result[i] = fp[j] + t * (fp[j + 1] - fp[j]);
            }
            return result;
        }

        /// <summary>
        /// interp1d(np.arange(n), y, kind='linear', fill_value='extrapolate') 在点 xs 处求值。
        /// xs 越界时线性外推。
        /// </summary>
        public static double LinearExtrap(double[] y, double x) {
            int n = y.Length;
            if (n == 1) return y[0];
            if (x <= 0) {
                double t = x;
                return y[0] + t * (y[1] - y[0]);
            }
            if (x >= n - 1) {
                double t = x - (n - 1);
                return y[n - 1] + t * (y[n - 1] - y[n - 2]);
            }
            int i = (int)Math.Floor(x);
            double frac = x - i;
            return y[i] + frac * (y[i + 1] - y[i]);
        }

        /// <summary>Python round()（银行家舍入）。</summary>
        public static double PyRound(double x) => Math.Round(x, MidpointRounding.ToEven);

        /// <summary>np.linspace(0, n-1, m)。</summary>
        public static double[] LinSpace01(int n, int m) {
            var xs = new double[m];
            if (m == 1) { if (n > 0) xs[0] = 0; return xs; }
            if (n == 1) {
                for (int i = 0; i < m; i++) xs[i] = 0;
                return xs;
            }
            double step = (n - 1) / (double)(m - 1);
            for (int i = 0; i < m; i++) xs[i] = step * i;
            return xs;
        }

        /// <summary>
        /// synthesis_pipeline.utils.interp_to_len：将一维数组线性插值到目标长度。
        /// 空/退化为常数的分支与 Python 版一致。
        /// </summary>
        public static double[] InterpToLen(double[] arr, int targetLen) {
            int n = arr.Length;
            if (n == targetLen) return (double[])arr.Clone();
            if (n == 0 || targetLen <= 0) {
                var constant = new double[Math.Max(targetLen, 0)];
                double fill = n > 0 ? arr[0] : 0;
                for (int i = 0; i < constant.Length; i++) constant[i] = fill;
                return constant;
            }
            var xs = LinSpace01(n, targetLen);
            var result = new double[targetLen];
            for (int i = 0; i < targetLen; i++) {
                result[i] = LinearExtrap(arr, xs[i]);
            }
            return result;
        }

        /// <summary>
        /// synthesis_pipeline.utils.resample_array：按 hop 比例重采样帧级参数。
        /// target_len = max(1, round(len * origHop/targetHop))（银行家舍入）。
        /// </summary>
        public static double[] ResampleArray(double[] arr, int origHop, int targetHop) {
            double ratio = (double)origHop / targetHop;
            int targetLen = Math.Max(1, (int)PyRound(arr.Length * ratio));
            return InterpToLen(arr, targetLen);
        }
    }
}
