using System;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// numpy legacy RandomState 的精确移植（MT19937 + rk_gauss），
    /// 用于 growl 效果的噪声序列复现（Python: np.random.RandomState(42).standard_normal(n)）。
    /// 参考 numpy/random/mtrand/randomkit.c。
    /// </summary>
    public sealed class NumpyRandom {
        private const int N = 624;
        private const int M = 397;
        private const uint MatrixA = 0x9908b0dfu;
        private const uint UpperMask = 0x80000000u;
        private const uint LowerMask = 0x7fffffffu;

        private readonly uint[] mt = new uint[N];
        private int mti = N + 1;   // 表示需要重新生成
        private bool hasGauss;
        private double gaussCache;

        public NumpyRandom(int seed) {
            // init_genrand
            mt[0] = (uint)(seed & 0xffffffffu);
            for (int i = 1; i < N; i++) {
                mt[i] = (uint)(1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint)i);
            }
            mti = N;
        }

        private uint NextUint32() {
            uint y;
            if (mti >= N) {
                int kk;
                for (kk = 0; kk < N - M; kk++) {
                    y = (mt[kk] & UpperMask) | (mt[kk + 1] & LowerMask);
                    mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1) != 0 ? MatrixA : 0u);
                }
                for (; kk < N - 1; kk++) {
                    y = (mt[kk] & UpperMask) | (mt[kk + 1] & LowerMask);
                    mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1) != 0 ? MatrixA : 0u);
                }
                y = (mt[N - 1] & UpperMask) | (mt[0] & LowerMask);
                mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1) != 0 ? MatrixA : 0u);
                mti = 0;
            }
            y = mt[mti++];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            return y;
        }

        /// <summary>rk_double：53 位精度均匀 [0,1)。</summary>
        private double NextDouble() {
            uint a = NextUint32() >> 5;
            uint b = NextUint32() >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        /// <summary>rk_gauss：极坐标法（带缓存），与 numpy standard_normal 逐值一致。</summary>
        public double StandardNormal() {
            if (hasGauss) {
                hasGauss = false;
                return gaussCache;
            }
            double x, y, r2;
            do {
                x = 2.0 * NextDouble() - 1.0;
                y = 2.0 * NextDouble() - 1.0;
                r2 = x * x + y * y;
            } while (r2 >= 1.0 || r2 == 0.0);
            double f = Math.Sqrt(-2.0 * Math.Log(r2) / r2);
            gaussCache = x * f;
            hasGauss = true;
            return y * f;
        }

        /// <summary>standard_normal(n)。</summary>
        public double[] StandardNormal(int n) {
            var result = new double[n];
            for (int i = 0; i < n; i++) result[i] = StandardNormal();
            return result;
        }
    }
}
