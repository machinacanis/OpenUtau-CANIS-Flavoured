using System;
using OpenUtau.Core.HiFiUtau.Engine.Dsp;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// engine.py _postprocess + post_process.py + tension_filter.py + warmth.py
    /// + growl.py + _numba_ops.py 的移植。数值语义逐行对照 Python 版。
    /// </summary>
    public static class PostProcessing {
        private const int StftNfft = 2048;
        private const int StftHop = 512;
        private static readonly float[] hannPeriodic = Stft.HannPeriodic(StftNfft);

        // scipy.signal.butter(4, 2000.0, fs=44100, output='sos') 的精确系数（常数）
        private static readonly double[][] SosLow = {
            new[] { 2.9136579221204529e-04, 5.8273158442409057e-04, 2.9136579221204529e-04, 1.0, -1.5236413003220306, 0.58766346876782993 },
            new[] { 1.0, 2.0, 1.0, 1.0, -1.7329280106367695, 0.80574423646235072 },
        };
        private static readonly double[][] SosHigh = {
            new[] { 0.6881179899153391, -1.3762359798306782, 0.6881179899153391, 1.0, -1.5236413003220306, 0.58766346876782993 },
            new[] { 1.0, -2.0, 1.0, 1.0, -1.7329280106367695, 0.80574423646235072 },
        };

        // ══════════════════ 主入口（engine.py _postprocess） ══════════════════

        public static byte[] Postprocess(SynthesisEngine engine, float[]? wav,
                float[]? harmonicIn, float[]? noiseIn, PhraseData phrase) {
            double[]? h = harmonicIn != null ? ToDouble(harmonicIn) : null;
            double[]? n = noiseIn != null ? ToDouble(noiseIn) : null;
            // engine.py: harmonic+noise 都有 → wav = harmonic + noise；否则基信号 = wav
            double[] baseWav = (h != null && n != null) ? Add(h, n) : ToDouble(wav!);

            int frontDh = (int)Interpolation.PyRound(engine.FrontPadFrames * (double)engine.ModelHop / phrase.HopSize);
            int tailDh = (int)Interpolation.PyRound(engine.TailPadFrames * (double)engine.ModelHop / phrase.HopSize);

            double[] pit = phrase.GetParam("pit", "pitd");
            double[] breath = phrase.GetParam("brec", "breath");
            double[] tension = phrase.GetParam("tenc", "tension");
            double[] voicing = phrase.GetParam("voic", "voicing");
            double[] brel = phrase.GetParam("brel", "bret_low");
            double[] breh = phrase.GetParam("breh", "bret_high");
            double[] warm = phrase.GetParam("warm", "warmth");
            double[] hcmp = phrase.GetParam("hcmp", "hcmp");
            double[] lowcut = phrase.GetParam("lowc", "lowcut");
            double[] growl = phrase.GetParam("gwl", "growl");

            bool needBreath = breath.Length > 0 && !AllClose(breath, 0, atol: 0.5);
            bool needTension = tension.Length > 0 && !AllClose(tension, 0, atol: 0.5);
            bool needVoicing = voicing.Length > 0 && !AllClose(voicing, 100, rtol: 0.05);
            bool needBrel = brel.Length > 0 && !AllClose(brel, 0, atol: 0.5);
            bool needBreh = breh.Length > 0 && !AllClose(breh, 0, atol: 0.5);
            bool needHcmp = hcmp.Length > 0 && !AllClose(hcmp, 0, atol: 0.5);
            bool needWarm = warm.Length > 0 && !AllClose(warm, 0, atol: 0.5);
            bool needLegacy = needBreath || needTension || needVoicing || needBrel || needBreh;

            if (needLegacy || needHcmp || needWarm) {
                if (h == null || n == null) {
                    var engineHnsep = engine.HnsepModel ?? throw new InvalidOperationException(
                        "当前合成使用了依赖 HN-SEP 的参数，但 HN-SEP 模型未加载。");
                    var (hh, nn) = engineHnsep.Separate(ToFloat(baseWav));
                    h = hh; n = nn;
                }

                if (needLegacy) {
                    var padB = PadEdge(breath, frontDh, tailDh);
                    var padT = PadEdge(tension, frontDh, tailDh);
                    var padV = PadEdge(voicing, frontDh, tailDh);
                    var padF0 = PadEdge(pit, frontDh, tailDh);
                    var padBrel = PadEdge(brel, frontDh, tailDh);
                    var padBreh = PadEdge(breh, frontDh, tailDh);
                    (h, n) = ApplyHnsepPostprocessComponents(h, n, padB, padT, padV, phrase.SampleRateConst,
                        f0Curve: padF0, brelArray: padBrel, brehArray: padBreh);
                }

                if (needWarm) {
                    double warmVal = Mean(warm);
                    h = ApplyWarmthEq(h, warmVal, phrase.SampleRateConst);
                }

                if (needHcmp) {
                    double hcmpVal = Mean(hcmp);
                    h = ApplyHarmonicCompression(h, hcmpVal, phrase.SampleRateConst);
                }

                baseWav = Add(h!, n!);
            }

            // 低切（F0 跟随 Butterworth 高通）
            if (lowcut.Length > 0 && !AllClose(lowcut, 0, atol: 0.5)) {
                var paddedLowcut = InterpToLenWithPad(PadEdge(lowcut, frontDh, tailDh), baseWav.Length);
                var padF0 = PadEdge(pit, frontDh, tailDh);
                baseWav = ApplyDynamicLowcut(baseWav, paddedLowcut, phrase.SampleRateConst, f0Curve: padF0);
            }

            // 咆哮（最后一步）
            if (growl.Length > 0) {
                baseWav = ApplyGrowl(baseWav,
                    PadEdge(growl, frontDh, tailDh),
                    phrase.SampleRateConst,
                    f0: PadEdge(pit, frontDh, tailDh),
                    f0Hop: phrase.HopSize);
            }

            // 统一裁剪首尾补帧
            int frontTrim = engine.FrontPadFrames * engine.ModelHop;
            int tailTrim = engine.TailPadFrames * engine.ModelHop;
            if (frontTrim > 0 && baseWav.Length > frontTrim) {
                baseWav = Slice(baseWav, frontTrim, baseWav.Length - frontTrim);
            }
            if (tailTrim > 0 && baseWav.Length > tailTrim) {
                baseWav = Slice(baseWav, 0, baseWav.Length - tailTrim);
            }

            // 首音素淡入 / 尾音素淡出
            var firstEnv = phrase.Phones[0];
            var lastEnv = phrase.Phones[^1];
            int sr = phrase.SampleRateConst;

            double p0x = firstEnv.P0.X, p0y = firstEnv.P0.Y;
            double p1x = firstEnv.P1.X, p1y = firstEnv.P1.Y;
            int fadeInLen = Math.Max(0, (int)Math.Round((p1x - p0x) / 1000 * sr));
            if (fadeInLen > 1 && baseWav.Length > fadeInLen && p0y < p1y) {
                double startGain = Math.Max(p0y / 100.0, 1e-6);
                for (int i = 0; i < fadeInLen; i++) {
                    double g = startGain + (p1y / 100.0 - startGain) * i / (fadeInLen - 1);
                    baseWav[i] *= g;
                }
            }
            double p3x = lastEnv.P3.X, p3y = lastEnv.P3.Y;
            double p4x = lastEnv.P4.X, p4y = lastEnv.P4.Y;
            int fadeOutLen = Math.Max(0, (int)Math.Round((p4x - p3x) / 1000 * sr));
            if (fadeOutLen > 1 && baseWav.Length > fadeOutLen && p3y > p4y) {
                int start = baseWav.Length - fadeOutLen;
                for (int i = 0; i < fadeOutLen; i++) {
                    double g = p3y / 100.0 + (p4y / 100.0 - p3y / 100.0) * i / (fadeOutLen - 1);
                    baseWav[start + i] *= g;
                }
            }

            var floats = ToFloat(baseWav);
            return SynthesisEngine.WavBytes(floats);
        }

        // ══════════════════ post_process.py ══════════════════

        /// <summary>apply_hnsep_postprocess_components。</summary>
        public static (double[] harmonic, double[] noise) ApplyHnsepPostprocessComponents(
                double[] harmonic, double[] noise,
                double[] breathArray, double[] tensionArray, double[] voicingArray,
                int sr, double[]? f0Curve = null,
                double[]? brelArray = null, double[]? brehArray = null) {
            int nSamples = harmonic.Length;

            if (breathArray.Length > 0 && !AllClose(breathArray, 0, atol: 0.5)) {
                var ratio = ResampleToLen(breathArray, nSamples);
                for (int i = 0; i < nSamples; i++) {
                    double r = ratio[i] / 100.0;
                    double gain = r > 0 ? 1.0 + r * 3.0 : 1.0 + r;
                    noise[i] *= Math.Clamp(gain, 0.0, 10.0);
                }
            }

            bool needBrel = brelArray != null && brelArray.Length > 0 && !AllClose(brelArray, 0, atol: 0.5);
            bool needBreh = brehArray != null && brehArray.Length > 0 && !AllClose(brehArray, 0, atol: 0.5);
            if (needBrel || needBreh) {
                var lowGain = Ones(nSamples);
                var highGain = Ones(nSamples);
                if (needBrel) {
                    var r = ResampleToLen(brelArray!, nSamples);
                    for (int i = 0; i < nSamples; i++) {
                        double v = r[i] / 100.0;
                        lowGain[i] = v > 0 ? 1.0 + v * 3.0 : 1.0 + v;
                    }
                }
                if (needBreh) {
                    var r = ResampleToLen(brehArray!, nSamples);
                    for (int i = 0; i < nSamples; i++) {
                        double v = r[i] / 100.0;
                        highGain[i] = v > 0 ? 1.0 + v * 3.0 : 1.0 + v;
                    }
                }
                noise = ApplyBreathBandGain(noise, lowGain, highGain, sr);
            }

            if (voicingArray.Length > 0 && !AllClose(voicingArray, 100, rtol: 0.05)) {
                var voicingMap = ResampleToLen(voicingArray, nSamples);
                for (int i = 0; i < nSamples; i++) {
                    double gain = Math.Clamp(voicingMap[i], 0, 500) / 100.0;
                    harmonic[i] = gain < 1e-8 ? 0.0 : harmonic[i] * gain;
                }
            }

            if (tensionArray.Length > 0 && !AllClose(tensionArray, 0, atol: 0.5)) {
                var tensionMap = ResampleToLen(tensionArray, nSamples);
                harmonic = ApplyDynamicTension(harmonic, tensionMap, sr, f0Curve: f0Curve);
            }

            return (harmonic, noise);
        }

        /// <summary>tension_filter.apply_breath_band_gain：分频后独立乘增益再相加。</summary>
        public static double[] ApplyBreathBandGain(double[] waveform, double[] lowGain, double[] highGain,
                int sr, double crossoverHz = 2000.0) {
            int n = waveform.Length;
            if (n == 0) return waveform;
            double lowMax = 0, highMax = 0;
            foreach (var v in lowGain) lowMax = Math.Max(lowMax, Math.Abs(v - 1.0));
            foreach (var v in highGain) highMax = Math.Max(highMax, Math.Abs(v - 1.0));
            if (lowMax < 0.01 && highMax < 0.01) return waveform;

            // 引擎内 crossover 固定 2000Hz/sr=44100（Python 缓存的常数系数）
            var sosLow = Math.Abs(crossoverHz - 2000.0) < 0.1 && sr == 44100 ? SosLow : null;
            var sosHigh = Math.Abs(crossoverHz - 2000.0) < 0.1 && sr == 44100 ? SosHigh : null;
            if (sosLow == null) throw new NotSupportedException("仅支持 crossover=2000Hz/sr=44100");

            var low = SosFilt(sosLow, waveform);
            var high = SosFilt(sosHigh, waveform);
            var result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = low[i] * lowGain[i] + high[i] * highGain[i];
            }
            return result;
        }

        /// <summary>scipy.signal.sosfilt（biquad 级联，a0 归一化）。</summary>
        public static double[] SosFilt(double[][] sos, double[] x) {
            int n = x.Length;
            var outArr = new double[n];
            // 各 section 状态
            int sections = sos.Length;
            var z = new double[sections, 2];
            for (int i = 0; i < n; i++) {
                double val = x[i];
                for (int s = 0; s < sections; s++) {
                    double b0 = sos[s][0] / sos[s][3], b1 = sos[s][1] / sos[s][3], b2 = sos[s][2] / sos[s][3];
                    double a1 = sos[s][4] / sos[s][3], a2 = sos[s][5] / sos[s][3];
                    double y = b0 * val + z[s, 0];
                    z[s, 0] = b1 * val - a1 * y + z[s, 1];
                    z[s, 1] = b2 * val - a2 * y;
                    val = y;
                }
                outArr[i] = val;
            }
            return outArr;
        }

        // ══════════════════ tension_filter.py ══════════════════

        /// <summary>apply_dynamic_tension：STFT 域逐帧频谱倾斜。</summary>
        public static double[] ApplyDynamicTension(double[] waveform, double[] tensionMap, int sr,
                double[]? f0Curve = null) {
            int originalLen = waveform.Length;
            int padLen = (StftHop - originalLen % StftHop) % StftHop;
            var padded = new double[originalLen + padLen];
            Array.Copy(waveform, padded, originalLen);

            var floats = ToFloat(padded);
            var (re, im, bins, frames) = Stft.Forward(floats, StftNfft, StftHop, hannPeriodic, center: true);
            int fftBin = bins;

            var tensionFrames = ResampleToLen(tensionMap, frames);

            var x0PerFrame = new double[frames];
            if (f0Curve != null && f0Curve.Length > 0) {
                var f0Frames = ResampleToLen(f0Curve, frames);
                for (int t = 0; t < frames; t++) {
                    double midpointHz = Math.Clamp(f0Frames[t] * 4.0, 400.0, 6000.0);
                    if (f0Frames[t] < 30.0) midpointHz = 1500.0;
                    x0PerFrame[t] = fftBin / ((sr / 2.0) / midpointHz);
                }
            } else {
                double x0Fixed = fftBin / ((sr / 2.0) / 1500);
                for (int t = 0; t < frames; t++) x0PerFrame[t] = x0Fixed;
            }

            // mag_db = log(clip(|D|,1e-9))
            var magDb = new double[bins * frames];
            var magLinear = new double[bins * frames];
            for (int i = 0; i < magDb.Length; i++) {
                double r = re[i], m = im[i];
                double mag = Math.Sqrt(r * r + m * m);
                magLinear[i] = mag;
                magDb[i] = Math.Log(Math.Max(mag, 1e-9));
            }

            ApplyTiltToFrames(magDb, magLinear, tensionFrames, x0PerFrame, frames, fftBin);

            // ISTFT：mag_out * e^{i·phase}
            var outRe = new float[bins * frames];
            var outIm = new float[bins * frames];
            for (int i = 0; i < outRe.Length; i++) {
                double magOut = Math.Exp(magDb[i]);
                double phase = Math.Atan2(im[i], re[i]);
                outRe[i] = (float)(magOut * Math.Cos(phase));
                outIm[i] = (float)(magOut * Math.Sin(phase));
            }
            var filtered = Stft.Inverse(outRe, outIm, StftNfft, StftHop, hannPeriodic, center: true);
            int outLen = Math.Min(originalLen, filtered.Length);
            var result = new double[outLen];
            for (int i = 0; i < outLen; i++) result[i] = filtered[i];

            // 软限幅
            double peak = 0;
            foreach (var v in result) peak = Math.Max(peak, Math.Abs(v));
            if (peak > 1.0) {
                for (int i = 0; i < result.Length; i++) result[i] = Math.Tanh(result[i] * 0.9) / 0.9;
                double newPeak = 0;
                foreach (var v in result) newPeak = Math.Max(newPeak, Math.Abs(v));
                if (newPeak > 1e-8) {
                    double scale = Math.Min(peak, 1.0) / newPeak;
                    for (int i = 0; i < result.Length; i++) result[i] *= scale;
                }
            }
            return result;
        }

        /// <summary>_apply_tilt_to_frames：log 幅度域逐帧倾斜滤波 + 能量保持。</summary>
        private static void ApplyTiltToFrames(double[] magDb, double[] magLinear,
                double[] tensionFrames, double[] x0PerFrame, int nFrames, int fftBin) {
            for (int t = 0; t < nFrames; t++) {
                double tv = tensionFrames[t];
                double b = tv > 0 ? -tv / 150.0 : -tv / 50.0;

                if (Math.Abs(b) < 0.001) continue;

                double x0 = x0PerFrame[t];
                if (x0 <= 0) x0 = 1.0;

                // 逐帧能量保持所需的原始总幅度
                double origSum = 0;
                for (int f = 0; f < fftBin; f++) origSum += magLinear[f * nFrames + t];

                for (int f = 0; f < fftBin; f++) {
                    double val = (-b / x0) * f + b;
                    if (val > 2.0) val = 2.0;
                    else if (val < -2.0) val = -2.0;
                    magDb[f * nFrames + t] += val;
                }

                if (origSum > 1e-12) {
                    double newSum = 0;
                    for (int f = 0; f < fftBin; f++) newSum += Math.Exp(magDb[f * nFrames + t]);
                    if (newSum > 1e-12) {
                        double comp = Math.Log(origSum / newSum);
                        for (int f = 0; f < fftBin; f++) magDb[f * nFrames + t] += comp;
                    }
                }

                if (b < -0.001) {
                    double val = b / (-15.0);
                    val = Math.Clamp(val, 0.0, 0.33);
                    double bGain = Math.Log(val + 1.0);
                    if (bGain > 0.0) {
                        for (int f = 0; f < fftBin; f++) magDb[f * nFrames + t] += bGain;
                    }
                }
            }
        }

        /// <summary>apply_dynamic_lowcut：STFT 域 Butterworth 高通（F0 跟随）。</summary>
        public static double[] ApplyDynamicLowcut(double[] waveform, double[] lowcutMap, int sr,
                double[]? f0Curve = null) {
            double lowcutMax = 0;
            foreach (var v in lowcutMap) lowcutMax = Math.Max(lowcutMax, v);
            if (lowcutMax < 0.5) return waveform;

            int originalLen = waveform.Length;
            int padLen = (StftHop - originalLen % StftHop) % StftHop;
            var padded = new double[originalLen + padLen];
            Array.Copy(waveform, padded, originalLen);
            var floats = ToFloat(padded);
            var (re, im, bins, frames) = Stft.Forward(floats, StftNfft, StftHop, hannPeriodic, center: true);
            int fftBin = bins;

            var lowcutFrames = ResampleToLen(lowcutMap, frames);
            var x0PerFrame = new double[frames];
            if (f0Curve != null && f0Curve.Length > 0) {
                var f0Frames = ResampleToLen(f0Curve, frames);
                for (int t = 0; t < frames; t++) {
                    double cutoffHz = (lowcutFrames[t] / 100.0) * f0Frames[t];
                    if (f0Frames[t] < 30.0) cutoffHz = 0.0;
                    cutoffHz = Math.Clamp(cutoffHz, 0.0, 2000.0);
                    x0PerFrame[t] = cutoffHz > 1.0 ? fftBin / ((sr / 2.0) / cutoffHz) : 0.0;
                }
            } else {
                for (int t = 0; t < frames; t++) {
                    double cutoffHz = Math.Clamp(lowcutFrames[t] * 5.0, 0.0, 500.0);
                    x0PerFrame[t] = cutoffHz > 1.0 ? fftBin / ((sr / 2.0) / cutoffHz) : 0.0;
                }
            }

            var magDb = new double[bins * frames];
            for (int i = 0; i < magDb.Length; i++) {
                double r = re[i], m = im[i];
                magDb[i] = Math.Log(Math.Max(Math.Sqrt(r * r + m * m), 1e-9));
            }

            for (int t = 0; t < frames; t++) {
                double cutoffBin = x0PerFrame[t];
                if (cutoffBin <= 0.5) continue;
                for (int f = 0; f < fftBin; f++) {
                    if (f == 0) {
                        magDb[0 * frames + t] -= 80.0;
                        continue;
                    }
                    double ratio = cutoffBin / f;
                    double r2 = ratio * ratio;
                    double gainDb = -10.0 * Math.Log10(1.0 + r2 * r2);
                    magDb[f * frames + t] += gainDb;
                }
            }

            var outRe = new float[bins * frames];
            var outIm = new float[bins * frames];
            for (int i = 0; i < outRe.Length; i++) {
                double magOut = Math.Exp(magDb[i]);
                double phase = Math.Atan2(im[i], re[i]);
                outRe[i] = (float)(magOut * Math.Cos(phase));
                outIm[i] = (float)(magOut * Math.Sin(phase));
            }
            var filtered = Stft.Inverse(outRe, outIm, StftNfft, StftHop, hannPeriodic, center: true);
            int outLen = Math.Min(originalLen, filtered.Length);
            var result = new double[outLen];
            for (int i = 0; i < outLen; i++) result[i] = filtered[i];
            return result;
        }

        // ══════════════════ warmth.py ══════════════════

        /// <summary>apply_warmth_eq：温暖/清凉通道条（压缩 + 饱和 + STFT EQ + 归一化）。</summary>
        public static double[] ApplyWarmthEq(double[] waveform, double warmthValue, int sr) {
            warmthValue = -warmthValue;  // OpenUtau 发送 warm=-100 表示温暖
            if (Math.Abs(warmthValue) < 0.5) return waveform;

            double warmth = warmthValue / 100.0;
            var x = (double[])waveform.Clone();

            if (warmth > 0) {
                double peakDb = 20.0 * Math.Log10(MaxAbs(x) + 1e-12);
                double threshold = peakDb - 18.0 + (1.0 - warmth) * 6.0;
                double ratio = 1.0 + warmth * 3.0;
                x = SoftKneeCompressor(x, threshold, ratio, sr);
                x = ApplySaturation(x, warmth * 0.3);
                x = ApplyStftEq(x, 300.0, warmth * 1.25, sigmaOct: 2.4, sr: sr);
            } else {
                double cool = -warmth;
                double peakDb = 20.0 * Math.Log10(MaxAbs(x) + 1e-12);
                double threshold = peakDb - 20.0;
                double ratio = 1.0 + cool * 1.0;
                x = SoftKneeCompressor(x, threshold, ratio, sr);
                x = ApplyStftEq(x, 5000.0, cool * 2.5, sigmaOct: 3.0, sr: sr);
            }

            // 能量归一化
            double rmsIn = Rms(waveform);
            double rmsOut = Rms(x);
            if (rmsOut > 1e-12 && rmsIn > 1e-12) {
                double gain = Math.Clamp(rmsIn / rmsOut, 0.01, 100.0);
                for (int i = 0; i < x.Length; i++) x[i] *= gain;
            }

            // 软限幅
            double peak = MaxAbs(x);
            if (peak > 0.99) {
                double scale = 0.99 / peak;
                for (int i = 0; i < x.Length; i++) x[i] *= scale;
            }
            return x;
        }

        /// <summary>apply_harmonic_compression：纯压缩 + 能量归一化。</summary>
        public static double[] ApplyHarmonicCompression(double[] waveform, double hcmpValue, int sr) {
            if (Math.Abs(hcmpValue) < 0.5) return waveform;
            double hcmp = hcmpValue / 100.0;
            var x = (double[])waveform.Clone();

            double peakDb = 20.0 * Math.Log10(MaxAbs(x) + 1e-12);
            double threshold = peakDb - 22.0;
            double ratio = 1.0 + hcmp * 19.0;
            x = SoftKneeCompressor(x, threshold, ratio, sr, attackMs: 1.0);

            double rmsIn = Rms(waveform);
            double rmsOut = Rms(x);
            if (rmsOut > 1e-12 && rmsIn > 1e-12) {
                double gain = Math.Clamp(rmsIn / rmsOut, 0.01, 100.0);
                for (int i = 0; i < x.Length; i++) x[i] *= gain;
            }

            double peak = MaxAbs(x);
            if (peak > 0.99) {
                double scale = 0.99 / peak;
                for (int i = 0; i < x.Length; i++) x[i] *= scale;
            }
            return x;
        }

        /// <summary>_apply_stft_eq：STFT 域钟形 EQ。</summary>
        public static double[] ApplyStftEq(double[] xArr, double fc, double gainDb,
                double sigmaOct = 2.0, int sr = 44100) {
            int originalLen = xArr.Length;
            int padLen = (StftHop - originalLen % StftHop) % StftHop;
            var padded = new double[originalLen + padLen];
            Array.Copy(xArr, padded, originalLen);
            var floats = ToFloat(padded);
            var (re, im, bins, frames) = Stft.Forward(floats, StftNfft, StftHop, hannPeriodic, center: true);

            // 钟形增益曲线（log2 频率域）
            var gainCurve = new double[bins];
            double logFc = Math.Log2(fc);
            for (int b = 0; b < bins; b++) {
                double freq = (double)b / (bins - 1) * (sr / 2.0);
                double logF = freq > 1.0 ? Math.Log2(freq) : Math.Log2(Math.Max(1.0 + 1e-9, 1.0));
                // Python: log_f[~mask] = log_f[mask][0]（第一个正频率的 log2）
                if (freq <= 1.0) logF = Math.Log2(sr / 2.0 / (bins - 1));
                double bell = Math.Exp(-0.5 * Math.Pow((logF - logFc) / sigmaOct, 2));
                double curve = 2.0 * bell - 1.0;
                if (curve < 0) curve = -Math.Pow(-curve, 0.7);
                gainCurve[b] = curve * gainDb;
            }

            var outRe = new float[bins * frames];
            var outIm = new float[bins * frames];
            for (int t = 0; t < frames; t++) {
                for (int b = 0; b < bins; b++) {
                    double r = re[b * frames + t], m = im[b * frames + t];
                    double mag = Math.Sqrt(r * r + m * m);
                    double phase = Math.Atan2(m, r);
                    double magOut = Math.Exp(Math.Log(Math.Max(mag, 1e-12)) + gainCurve[b]);
                    outRe[b * frames + t] = (float)(magOut * Math.Cos(phase));
                    outIm[b * frames + t] = (float)(magOut * Math.Sin(phase));
                }
            }
            var filtered = Stft.Inverse(outRe, outIm, StftNfft, StftHop, hannPeriodic, center: true);
            int outLen = Math.Min(originalLen, filtered.Length);
            var result = new double[outLen];
            for (int i = 0; i < outLen; i++) result[i] = filtered[i];
            return result;
        }

        // ══════════════════ _numba_ops.py ══════════════════

        /// <summary>soft_knee_compressor：RMS 软膝压缩器。</summary>
        public static double[] SoftKneeCompressor(double[] signal, double thresholdDb, double ratio,
                int sr, double kneeDb = 6.0, double attackMs = 5.0, double releaseMs = 60.0) {
            int n = signal.Length;
            if (n == 0) return Array.Empty<double>();

            double rmsWin = Math.Max(1, (int)(sr * 0.01));
            int half = (int)(rmsWin / 2);

            // 反射填充 x²
            var padded = new double[n + 2 * half];
            for (int i = 0; i < half; i++) padded[i] = Sq(signal[half - 1 - i]);
            for (int i = 0; i < n; i++) padded[half + i] = Sq(signal[i]);
            for (int i = 0; i < half; i++) padded[half + n + i] = Sq(signal[n - 1 - i]);

            var rms = new double[n];
            double sumSq = 0;
            for (int i = 0; i < (int)Math.Min(rmsWin, padded.Length); i++) sumSq += padded[i];
            double invWin = 1.0 / rmsWin;
            for (int i = 0; i < n; i++) {
                rms[i] = Math.Sqrt(sumSq * invWin);
                if (i + (int)rmsWin < padded.Length) {
                    sumSq += padded[i + (int)rmsWin] - padded[i];
                }
            }

            double halfKnee = kneeDb / 2.0;
            double oneMinusInvRatio = 1.0 - 1.0 / ratio;
            double alphaA = attackMs > 0 ? Math.Exp(-1.0 / (sr * attackMs / 1000.0)) : 0.0;
            double alphaR = releaseMs > 0 ? Math.Exp(-1.0 / (sr * releaseMs / 1000.0)) : 0.0;

            var outArr = new double[n];
            double prevG = 1.0;
            for (int i = 0; i < n; i++) {
                double rd = 20.0 * Math.Log10(Math.Max(rms[i], 1e-12));
                double os = rd - thresholdDb;
                double gDb;
                if (os > halfKnee) {
                    gDb = -os * oneMinusInvRatio;
                } else if (os > -halfKnee) {
                    double ko = os + halfKnee;
                    gDb = (ko * ko) / (2.0 * kneeDb) * oneMinusInvRatio;
                } else {
                    gDb = 0.0;
                }
                double g = Math.Pow(10.0, gDb / 20.0);
                double gSmooth = g < prevG
                    ? alphaA * prevG + (1.0 - alphaA) * g
                    : alphaR * prevG + (1.0 - alphaR) * g;
                prevG = gSmooth;
                outArr[i] = signal[i] * gSmooth;
            }
            return outArr;
        }

        /// <summary>apply_saturation：电子管式谐波饱和。</summary>
        public static double[] ApplySaturation(double[] signal, double drive) {
            int n = signal.Length;
            if (drive < 0.01 || n == 0) return (double[])signal.Clone();

            double peakIn = 0;
            foreach (var v in signal) peakIn = Math.Max(peakIn, Math.Abs(v));
            if (peakIn < 1e-12) return (double[])signal.Clone();

            double invPeak = 1.0 / peakIn;
            double driveGain = 1.0 + drive * 8.0;
            double bias = 0.05 * drive;
            double tanhBias = Math.Tanh(bias);
            double wet = drive * 0.5;
            double dry = 1.0 - wet;

            var outArr = new double[n];
            for (int i = 0; i < n; i++) {
                double val = signal[i] * invPeak * driveGain;
                val = Math.Tanh(val + bias) - tanhBias;
                val = Math.Tanh(val * 1.5);
                outArr[i] = (dry * (signal[i] * invPeak) + wet * val) * peakIn;
            }
            return outArr;
        }

        // ══════════════════ growl.py ══════════════════

        /// <summary>apply_growl：时域延迟调制咆哮效果（噪声序列与 numpy RandomState(42) 逐值一致）。</summary>
        public static double[] ApplyGrowl(double[] waveform, double[] growlArray, int sr,
                double baseFreq = 120.0, double[]? f0 = null, int? f0Hop = null) {
            int nSamples = waveform.Length;
            if (growlArray.Length == 0 || AllClose(growlArray, 0, atol: 0.5)) return waveform;

            var growlMap = ResampleToLen(growlArray, nSamples);

            // 每样本颤音频率（跟随音高，限 80~240Hz）
            var sine = new double[nSamples];
            var freq = new double[nSamples];
            if (f0 != null && f0.Length > 0 && f0Hop.HasValue) {
                var f0Map = ResampleToLen(f0, nSamples);
                double phase = 0;
                double dt = 1.0 / sr;
                for (int i = 0; i < nSamples; i++) {
                    freq[i] = Math.Clamp(f0Map[i] * 0.35, 80.0, 240.0);
                    phase += 2.0 * Math.PI * freq[i] * dt;
                    sine[i] = Math.Sin(phase);
                }
            } else {
                for (int i = 0; i < nSamples; i++) {
                    freq[i] = baseFreq;
                    sine[i] = Math.Sin(2.0 * Math.PI * baseFreq * i / (double)sr);
                }
            }

            const double maxDelaySec = 0.00012;
            var modulator = new double[nSamples];
            var rng = new NumpyRandom(42);   // 与 numpy RandomState(42) 逐值一致
            var rawNoise = rng.StandardNormal(nSamples);
            // 低通（滑动均值）
            int kernelSize = Math.Max(1, (int)(sr / 300));
            var noise = MovingAverage(rawNoise, kernelSize);
            double noiseMax = 0;
            foreach (var v in noise) noiseMax = Math.Max(noiseMax, Math.Abs(v));
            double noiseScale = 1.0 / (noiseMax + 1e-8);
            double modPeak = 0;
            for (int i = 0; i < nSamples; i++) {
                modulator[i] = 0.88 * sine[i] + 0.12 * noise[i] * noiseScale;
                modPeak = Math.Max(modPeak, Math.Abs(modulator[i]));
            }
            if (modPeak > 1e-8) {
                for (int i = 0; i < nSamples; i++) modulator[i] /= modPeak;
            }

            var result = new double[nSamples];
            double xMax = (nSamples - 1) / (double)sr;
            for (int i = 0; i < nSamples; i++) {
                double depth = Math.Pow(growlMap[i] / 100.0, 0.8);
                double t = i / (double)sr;
                double tau = Math.Clamp(t + depth * maxDelaySec * modulator[i], 0.0, xMax);
                // np.interp（端点截断，原始波形域）
                double pos = tau * sr;
                result[i] = NpInterpSample(pos, waveform);
            }

            double peak = 0;
            foreach (var v in result) peak = Math.Max(peak, Math.Abs(v));
            if (peak > 1.0) {
                double scale = 1.0 / peak;
                for (int i = 0; i < result.Length; i++) result[i] *= scale;
            }
            return result;
        }

        private static double NpInterpSample(double pos, double[] waveform) {
            if (pos <= 0) return waveform[0];
            if (pos >= waveform.Length - 1) return waveform[^1];
            int i = (int)pos;
            double frac = pos - i;
            return waveform[i] + frac * (waveform[i + 1] - waveform[i]);
        }

        private static double[] MovingAverage(double[] x, int kernelSize) {
            // np.convolve(x, ones(k)/k, 'same')
            int n = x.Length;
            var result = new double[n];
            int half = kernelSize / 2;
            for (int i = 0; i < n; i++) {
                double sum = 0;
                int count = 0;
                for (int j = 0; j < kernelSize; j++) {
                    int idx = i - half + j;
                    if (idx >= 0 && idx < n) { sum += x[idx]; count++; }
                }
                result[i] = count > 0 ? sum / kernelSize : 0;
            }
            return result;
        }

        // ══════════════════ 小工具 ══════════════════

        private static double[] ToDouble(float[] x) {
            var d = new double[x.Length];
            for (int i = 0; i < x.Length; i++) d[i] = x[i];
            return d;
        }

        private static float[] ToFloat(double[] x) {
            var f = new float[x.Length];
            for (int i = 0; i < x.Length; i++) f[i] = (float)x[i];
            return f;
        }

        private static double[] Add(double[] a, double[] b) {
            int n = Math.Min(a.Length, b.Length);
            var result = new double[n];
            for (int i = 0; i < n; i++) result[i] = a[i] + b[i];
            return result;
        }

        private static double[] Ones(int n) {
            var r = new double[n];
            for (int i = 0; i < n; i++) r[i] = 1.0;
            return r;
        }

        private static double Mean(double[] a) {
            double s = 0;
            foreach (var v in a) s += v;
            return s / a.Length;
        }

        private static double MaxAbs(double[] a) {
            double m = 0;
            foreach (var v in a) m = Math.Max(m, Math.Abs(v));
            return m;
        }

        private static double Rms(double[] a) {
            double s = 0;
            foreach (var v in a) s += v * v;
            return Math.Sqrt(s / a.Length);
        }

        private static double Sq(double v) => v * v;

        private static double[] Slice(double[] a, int start, int len) {
            var r = new double[len];
            Array.Copy(a, start, r, 0, len);
            return r;
        }

        /// <summary>np.allclose 语义（atol/rtol 均相对参数或 1）。</summary>
        private static bool AllClose(double[] arr, double target, double atol = 1e-8, double rtol = 1e-5) {
            foreach (var v in arr) {
                if (Math.Abs(v - target) > atol + rtol * Math.Abs(target)) return false;
            }
            return true;
        }

        /// <summary>engine.py _pad：首尾以首/末值延拓。</summary>
        private static double[] PadEdge(double[] arr, int front, int tail) {
            if (arr.Length == 0) return arr;
            var result = new double[front + arr.Length + tail];
            for (int i = 0; i < front; i++) result[i] = arr[0];
            Array.Copy(arr, 0, result, front, arr.Length);
            for (int i = 0; i < tail; i++) result[front + arr.Length + i] = arr[^1];
            return result;
        }

        /// <summary>interp_to_len 包装（ResampleToLen 与 Python interp_to_len 等价）。</summary>
        private static double[] ResampleToLen(double[] arr, int targetLen)
            => Interpolation.InterpToLen(arr, targetLen);

        private static double[] InterpToLenWithPad(double[] arr, int targetLen)
            => Interpolation.InterpToLen(arr, targetLen);
    }
}
