using System;
using System.Numerics;

namespace OpenUtau.Core.HiFiUtau.Engine.Dsp {
    /// <summary>
    /// mel 投影用的 SIMD 点积（替代 librosa/numpy 的 BLAS matmul）。
    /// 输入两侧均须为连续 float32 段；与标量 double 版差异 ~1e-5，远小于金标准阈值。
    /// </summary>
    internal static class Simd {
        public static float Dot(float[] a, int aOff, float[] b, int bOff, int n) {
            int vecEnd = n - n % Vector<float>.Count;
            var acc = Vector<float>.Zero;
            for (int i = 0; i < vecEnd; i += Vector<float>.Count) {
                acc += new Vector<float>(a, aOff + i) * new Vector<float>(b, bOff + i);
            }
            float sum = 0;
            for (int i = 0; i < Vector<float>.Count; i++) {
                sum += acc[i];
            }
            for (int i = vecEnd; i < n; i++) {
                sum += a[aOff + i] * b[bOff + i];
            }
            return sum;
        }

        public static float[] ToFloat32(double[] x) {
            var r = new float[x.Length];
            for (int i = 0; i < x.Length; i++) r[i] = (float)x[i];
            return r;
        }
    }
}
