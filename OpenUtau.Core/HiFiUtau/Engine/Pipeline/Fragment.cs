using System;
using System.Threading.Tasks;
using OpenUtau.Core.HiFiUtau.Engine.Dsp;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// synthesis_pipeline/fragment.py 的移植（SPLC=0 feat 管线）。
    /// hop=64 提取 mel，含上下文填充、时间拉伸、strt 循环、左侧裁剪/补空白、P 归一化、
    /// phtp 音量匹配、genc/shft 频域偏移。数值语义逐行对照 Python 版。
    /// </summary>
    public static class Fragment {
        public const int SampleRate = 44100;
        public const int HopLength = 64;
        public const int Nfft = 2048;
        public const int WinLength = 2048;
        public const int NMels = 128;
        public const double MsPerFrame = (double)HopLength / SampleRate * 1000;

        private static readonly double[] melBasis = MelFilterbank.Basis(SampleRate, Nfft, NMels, 40, 16000);
        private static readonly float[] melBasisF32 = Simd.ToFloat32(melBasis);
        private static readonly float[] hannPeriodic = Stft.HannPeriodic(WinLength);

        /// <summary>fragment.py _audio_to_mel：log(clip(melspectrogram(power=1), 1e-5))。</summary>
        public static (double[] mel, int frames) AudioToMel(float[] audio) {
            if (audio.Length == 0) return (Array.Empty<double>(), 0);
            var (re, im, bins, frames) = Stft.Forward(audio, Nfft, HopLength, hannPeriodic, center: true);
            // 幅度谱转置为帧主序 [t*bins+b]，mel 投影两侧均连续（SIMD 点积 + 按频带并行）
            var magT = new float[bins * frames];
            for (int t = 0; t < frames; t++) {
                int src = t, dst = t * bins;
                for (int b = 0; b < bins; b++) {
                    float r = re[src], m = im[src];
                    magT[dst + b] = MathF.Sqrt(r * r + m * m);
                    src += frames;
                }
            }
            var mel = new double[NMels * frames];
            Parallel.For(0, NMels, m => {
                int bOff = m * bins, oOff = m * frames;
                for (int t = 0; t < frames; t++) {
                    float sum = Simd.Dot(melBasisF32, bOff, magT, t * bins, bins);
                    mel[oOff + t] = Math.Log(Math.Max(sum, 1e-5));
                }
            });
            return (mel, frames);
        }

        /// <summary>fragment.py cut_audio（串行版）。</summary>
        public static void CutAudio(PhraseData phrase) {
            foreach (var info in phrase.Phones) {
                ProcessSinglePhoneme(phrase, info);
            }
        }

        /// <summary>fragment.py _process_single_phoneme。</summary>
        public static void ProcessSinglePhoneme(PhraseData phrase, PhoneInfo info) {
            string wavPath = info.Oto.AudioFilePath;
            var audio = AudioReader.Read(wavPath, SampleRate);
            double totalLenMs = audio.Length / (double)SampleRate * 1000;

            double offsetMs = info.Oto.Offset;
            double consonantMs = info.Oto.Consonant;
            double cutoffMs = info.Oto.Cutoff;
            double preutterMs = info.Oto.Preutter;

            double p0X = info.P0.X;
            double vel = info.RequireFlag("vel");
            double stretchFactor = Math.Pow(2.0, (100 - vel) / 100.0);
            double stretchedPreutter = preutterMs * stretchFactor;
            double preToLeftMs = stretchedPreutter + p0X;

            int startSample = (int)Math.Round(offsetMs / 1000 * SampleRate);
            int consonantSample = (int)Math.Round((offsetMs + consonantMs) / 1000 * SampleRate);
            int endSample = cutoffMs > 0
                ? (int)Math.Round((totalLenMs - cutoffMs) / 1000 * SampleRate)
                : (int)Math.Round((offsetMs + Math.Abs(cutoffMs)) / 1000 * SampleRate);

            startSample = Math.Max(0, Math.Min(startSample, audio.Length));
            consonantSample = Math.Max(startSample, Math.Min(consonantSample, audio.Length));
            endSample = Math.Max(consonantSample, Math.Min(endSample, audio.Length));

            int segLen = endSample - startSample;
            var audioSeg = new float[segLen];
            Array.Copy(audio, startSample, audioSeg, 0, segLen);
            int conSamples = consonantSample - startSample;

            // mel 提取：读更长音频留 STFT 上下文，再裁掉边缘帧
            int padContext = (int)Interpolation.PyRound((double)(Nfft / 2) / HopLength) * HopLength;
            int padFront = Math.Min(padContext, startSample);
            int padTail = Math.Min(padContext, audio.Length - endSample);

            int extLen = segLen + padFront + padTail;
            var audioExt = new float[extLen];
            Array.Copy(audio, startSample - padFront, audioExt, 0, extLen);
            var (melExt, melExtFrames) = AudioToMel(audioExt);

            int cropFront = padFront / HopLength;
            int cropTail = padTail / HopLength;
            int nFrames;
            double[] melFull;
            if (cropTail > 0) {
                nFrames = melExtFrames - cropFront - cropTail;
                melFull = SliceFrames(melExt, NMels, melExtFrames, cropFront, nFrames);
            } else {
                nFrames = melExtFrames - cropFront;
                melFull = SliceFrames(melExt, NMels, melExtFrames, cropFront, nFrames);
            }

            if (nFrames <= 0) {
                info.Mel = Array.Empty<double>();
                info.MelFrames = 0;
                info.AudioSeg = audioSeg;
                info.ConsonantFrames = 0;
                info.StretchFactor = 1.0;
                return;
            }

            // 辅音/元音帧数
            int conFramesOrig = conSamples > 0
                ? Math.Max(1, (int)((conSamples - HopLength) / (double)HopLength) + 1)
                : 0;
            conFramesOrig = Math.Min(conFramesOrig, nFrames);
            int vowFramesOrig = nFrames - conFramesOrig;

            double p4X = info.P4.X;
            double totalBudgetMs = p4X + stretchedPreutter;
            int totalBudgetFrames = Math.Max((int)(totalBudgetMs / MsPerFrame), 1);

            int targetConFrames = Math.Max(1, (int)(conFramesOrig * stretchFactor));
            targetConFrames = Math.Min(targetConFrames, totalBudgetFrames - 1);
            int targetVowFrames = totalBudgetFrames - targetConFrames;
            int totalFrames = totalBudgetFrames;

            // strt=1: reflect padding 正反循环
            int strt = (int)info.Flag("strt", 0);
            if (strt == 1 && vowFramesOrig > 1 && targetVowFrames > vowFramesOrig * 1.5) {
                int conFrames = conFramesOrig;
                int padExtra = Math.Min(4, vowFramesOrig / 2);
                int padFrames = targetVowFrames - vowFramesOrig + padExtra;
                // np.pad(mel_vowel, ((0,0),(0,pad)), 'reflect') 逐行反射
                var extended = new double[NMels * (conFrames + vowFramesOrig + padFrames)];
                for (int m = 0; m < NMels; m++) {
                    int srcOff = m * nFrames;
                    int dstOff = m * (conFrames + vowFramesOrig + padFrames);
                    Array.Copy(melFull, srcOff, extended, dstOff, conFrames + vowFramesOrig);
                    for (int k = 0; k < padFrames; k++) {
                        int srcIdx = ReflectFrameIndex(vowFramesOrig - 2 - k, vowFramesOrig);
                        extended[dstOff + conFrames + vowFramesOrig + k] = melFull[srcOff + conFrames + srcIdx];
                    }
                }
                melFull = extended;
                nFrames = conFrames + vowFramesOrig + padFrames;
                vowFramesOrig = nFrames - conFrames;
            }

            // 帧索引映射
            var mapped = new double[totalFrames];
            for (int t = 0; t < totalFrames; t++) {
                if (t < targetConFrames) {
                    mapped[t] = (double)t / stretchFactor;
                } else if (vowFramesOrig > 0) {
                    mapped[t] = conFramesOrig + (double)(t - targetConFrames) * (vowFramesOrig / (double)Math.Max(1, targetVowFrames));
                } else {
                    mapped[t] = conFramesOrig;
                }
                mapped[t] = Math.Clamp(mapped[t], 0, nFrames - 1);
            }

            // 线性拉伸插值（逐 band 沿帧轴）
            double[] melOut = InterpRows(melFull, NMels, nFrames, mapped);
            int outFrames = totalFrames;

            // strt=1: 抹平循环段能量起伏
            if (strt == 1 && targetVowFrames > 1 && outFrames >= targetConFrames + 2) {
                int vowStart = targetConFrames;
                int vowLen = outFrames - vowStart;
                var frameEnergy = new double[vowLen];
                for (int t = 0; t < vowLen; t++) {
                    double sum = 0;
                    for (int m = 0; m < NMels; m++) {
                        sum += Math.Exp(melOut[m * outFrames + vowStart + t]);
                    }
                    frameEnergy[t] = Math.Max(sum / NMels, 1e-12);
                }
                double e0 = frameEnergy[0], e1 = frameEnergy[vowLen - 1];
                for (int t = 0; t < vowLen; t++) {
                    double target = e0 + (e1 - e0) * t / (vowLen - 1);
                    double logT = Math.Log(target);
                    for (int m = 0; m < NMels; m++) {
                        melOut[m * outFrames + vowStart + t] += -Math.Log(frameEnergy[t]) + logT;
                    }
                }
            }

            int leftPadFrames = 0;
            // 左侧裁剪/补空白
            if (preToLeftMs > 0) {
                int leftCutFrames = (int)(preToLeftMs / MsPerFrame);
                if (leftCutFrames < outFrames) {
                    melOut = SliceFrames(melOut, NMels, outFrames, leftCutFrames, outFrames - leftCutFrames);
                    outFrames -= leftCutFrames;
                    targetConFrames = Math.Max(0, targetConFrames - leftCutFrames);
                } else {
                    melOut = Array.Empty<double>();
                    outFrames = 0;
                    targetConFrames = 0;
                }
            } else if (preToLeftMs < 0) {
                leftPadFrames = (int)(-preToLeftMs / MsPerFrame);
                var blank = new double[NMels * leftPadFrames];
                for (int i = 0; i < blank.Length; i++) blank[i] = Math.Log(1e-5);
                if (outFrames > 0) {
                    int fadeIn = Math.Min(12, leftPadFrames);
                    // 逐频带独立淡入，保留频谱形状
                    for (int t = 0; t < fadeIn; t++) {
                        double alpha = (t + 1) / (double)(fadeIn + 1);
                        for (int m = 0; m < NMels; m++) {
                            double first = Math.Exp(melOut[m * outFrames]);
                            blank[m * leftPadFrames + leftPadFrames - fadeIn + t] =
                                Math.Log(first * alpha + 1e-10);
                        }
                    }
                }
                var combined = new double[NMels * (leftPadFrames + outFrames)];
                Array.Copy(blank, combined, NMels * leftPadFrames);
                if (outFrames > 0) Array.Copy(melOut, 0, combined, NMels * leftPadFrames, NMels * outFrames);
                melOut = combined;
                outFrames += leftPadFrames;
                targetConFrames += leftPadFrames;
            }

            // P 参数：音量均衡
            double pFlag = info.Flag("P", 0);
            if (pFlag > 0 && outFrames > 0) {
                double targetRms = 0.5;
                int audioStart = preToLeftMs < 0 ? leftPadFrames : 0;
                if (outFrames > audioStart) {
                    int audioFrames = outFrames - audioStart;
                    double sumSq = 0;
                    for (int m = 0; m < NMels; m++) {
                        for (int t = 0; t < audioFrames; t++) {
                            double v = Math.Exp(melOut[m * outFrames + audioStart + t]);
                            sumSq += v * v;
                        }
                    }
                    double curRms = Math.Sqrt(sumSq / (NMels * audioFrames));
                    if (curRms > 1e-12) {
                        double blend = pFlag / 100.0;
                        double targetRmsActual = curRms * (1 - blend) + targetRms * blend;
                        double scale = targetRmsActual / curRms;
                        double logScale = Math.Log(scale);
                        for (int i = 0; i < melOut.Length; i++) melOut[i] += logScale;
                    }
                }
            }

            info.Mel = melOut;
            info.MelFrames = outFrames;
            info.AudioSeg = audioSeg;
            info.ConsonantFrames = targetConFrames;
            info.StretchFactor = stretchFactor;
        }

        private static int ReflectFrameIndex(int idx, int n) {
            if (n == 1) return 0;
            int span = 2 * (n - 1);
            int v = idx % span;
            if (v < 0) v += span;
            return v < n - 1 ? v : span - v;
        }

        /// <summary>band-major (rows, srcFrames) 的帧切片。</summary>
        public static double[] SliceFrames(double[] src, int rows, int srcFrames, int startFrame, int count) {
            if (count <= 0) return Array.Empty<double>();
            var dst = new double[rows * count];
            for (int r = 0; r < rows; r++) {
                Array.Copy(src, r * srcFrames + startFrame, dst, r * count, count);
            }
            return dst;
        }

        /// <summary>interp1d(axis=1) 等价：逐 band 沿帧轴对 newX 求值（含外推）。</summary>
        public static double[] InterpRows(double[] src, int rows, int srcFrames, double[] newX) {
            var dst = new double[rows * newX.Length];
            for (int r = 0; r < rows; r++) {
                int off = r * srcFrames;
                var row = new double[srcFrames];
                Array.Copy(src, off, row, 0, srcFrames);
                int dstOff = r * newX.Length;
                for (int i = 0; i < newX.Length; i++) {
                    dst[dstOff + i] = Interpolation.LinearExtrap(row, newX[i]);
                }
            }
            return dst;
        }

        /// <summary>fragment.py adjust_volume_by_phtp。msPerFrame 由管线决定（hop64/hop512）。</summary>
        public static void AdjustVolumeByPhtp(PhraseData phrase, double msPerFrame) {
            var phones = phrase.Phones;
            for (int i = 0; i < phones.Count; i++) {
                var info = phones[i];
                double phtp = info.Flag("phtp", 0);
                if (phtp == 0) continue;
                var melCur = info.Mel;
                if (melCur == null || info.MelFrames == 0) continue;

                if (phtp == 1 && i < phones.Count - 1) {
                    var nextInfo = phones[i + 1];
                    var melNext = nextInfo.Mel;
                    if (melNext == null || nextInfo.MelFrames == 0) continue;
                    double p0X = nextInfo.P0.X, p1X = nextInfo.P1.X;
                    double overlapMs = p1X < 0 ? Math.Abs(p0X) - Math.Abs(p1X) : Math.Abs(p1X) + Math.Abs(p0X);
                    if (overlapMs <= 0) continue;
                    int overlapFrames = (int)Interpolation.PyRound(overlapMs / msPerFrame);
                    int curTail = Math.Min(overlapFrames, info.MelFrames);
                    int nextHead = Math.Min(overlapFrames, nextInfo.MelFrames);
                    if (curTail <= 0 || nextHead <= 0) continue;
                    double rmsCur = MelRms(melCur, info.MelFrames, info.MelFrames - curTail, curTail);
                    double rmsNext = MelRms(melNext, nextInfo.MelFrames, 0, nextHead);
                    if (rmsCur < 1e-12 || rmsNext < 1e-12) continue;
                    double scale = rmsNext / rmsCur;
                    double logScale = Math.Log(scale);
                    for (int k = 0; k < melCur.Length; k++) melCur[k] += logScale;
                } else if (phtp == 2 && i > 0) {
                    var prevInfo = phones[i - 1];
                    var melPrev = prevInfo.Mel;
                    if (melPrev == null || prevInfo.MelFrames == 0) continue;
                    double p0X = info.P0.X, p1X = info.P1.X;
                    double overlapMs = p1X < 0 ? Math.Abs(p0X) - Math.Abs(p1X) : Math.Abs(p1X) + Math.Abs(p0X);
                    if (overlapMs <= 0) continue;
                    int overlapFrames = (int)Interpolation.PyRound(overlapMs / msPerFrame);
                    int prevTail = Math.Min(overlapFrames, prevInfo.MelFrames);
                    int curHead = Math.Min(overlapFrames, info.MelFrames);
                    if (prevTail <= 0 || curHead <= 0) continue;
                    double rmsPrev = MelRms(melPrev, prevInfo.MelFrames, prevInfo.MelFrames - prevTail, prevTail);
                    double rmsCur = MelRms(melCur, info.MelFrames, 0, curHead);
                    if (rmsPrev < 1e-12 || rmsCur < 1e-12) continue;
                    double scale = rmsPrev / rmsCur;
                    double logScale = Math.Log(scale);
                    for (int k = 0; k < melCur.Length; k++) melCur[k] += logScale;
                }
            }
        }

        private static double MelRms(double[] mel, int frames, int startFrame, int count) {
            double sumSq = 0;
            int n = NMels * count;
            for (int m = 0; m < NMels; m++) {
                for (int t = 0; t < count; t++) {
                    double v = Math.Exp(mel[m * frames + startFrame + t]);
                    sumSq += v * v;
                }
            }
            return Math.Sqrt(sumSq / n);
        }

        /// <summary>fragment.py apply_dynamic_gen_to_mels：音素级 genc/shft 频域偏移。</summary>
        public static void ApplyDynamicGenToMels(PhraseData phrase) {
            foreach (var info in phrase.Phones) {
                var mel = info.Mel;
                if (mel == null || info.MelFrames == 0) continue;
                int nMels = NMels;
                int T = info.MelFrames;

                var phGenc = info.GetPhoneDP("genc");
                bool useGenc = phGenc.Length > 0;

                if (useGenc) {
                    var gencInterp = Interpolation.InterpToLen(phGenc, T);
                    var melOut = new double[mel.Length];
                    var oldIdx = new double[nMels];
                    for (int m = 0; m < nMels; m++) oldIdx[m] = m;
                    for (int t = 0; t < T; t++) {
                        double factor = Math.Pow(2.0, (gencInterp[t] / 100.0) / 12.0);
                        if (Math.Abs(factor - 1.0) < 0.001) {
                            for (int m = 0; m < nMels; m++) melOut[m * T + t] = mel[m * T + t];
                        } else {
                            for (int m = 0; m < nMels; m++) {
                                double idx = Math.Clamp(oldIdx[m] / factor, 0, nMels - 1);
                                melOut[m * T + t] = NpInterpColumn(mel, nMels, T, t, idx);
                            }
                        }
                    }
                    info.Mel = melOut;
                    continue;
                }

                double shftVal = info.Flag("shft", 0);
                if (shftVal == 0) continue;
                shftVal = Math.Clamp(shftVal, -200, 200);
                double factorConst = Math.Pow(2.0, (shftVal / 100.0) / 12.0);
                if (Math.Abs(factorConst - 1.0) < 0.001) continue;
                var warped = new double[mel.Length];
                for (int t = 0; t < T; t++) {
                    for (int m = 0; m < nMels; m++) {
                        double idx = Math.Clamp(m / factorConst, 0, nMels - 1);
                        warped[m * T + t] = NpInterpColumn(mel, nMels, T, t, idx);
                    }
                }
                info.Mel = warped;
            }
        }

        /// <summary>np.interp(idx, arange(nMels), mel[:, t])：跨频带插值（端点截断）。</summary>
        private static double NpInterpColumn(double[] mel, int nMels, int T, int t, double idx) {
            if (idx <= 0) return mel[0 * T + t];
            if (idx >= nMels - 1) return mel[(nMels - 1) * T + t];
            int i = (int)idx;
            double frac = idx - i;
            return mel[i * T + t] + frac * (mel[(i + 1) * T + t] - mel[i * T + t]);
        }
    }
}
