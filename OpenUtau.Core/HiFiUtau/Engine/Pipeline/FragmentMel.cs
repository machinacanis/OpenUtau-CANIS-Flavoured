using System;
using OpenUtau.Core.HiFiUtau.Engine.Dsp;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// synthesis_pipeline/fragment_mel.py 的移植（SPLC=1 mel 管线）。
    /// hop=512 直接工作，用 MelExtractor 做自定义帧定位 mel 提取。
    /// </summary>
    public static class FragmentMel {
        public const int HopLength = 512;
        public const double MsPerFrame = (double)HopLength / 44100 * 1000;

        public static void CutAudio(PhraseData phrase, MelExtractor melExc) {
            var phones = phrase.Phones;
            int n = phones.Count;
            var (starts, preutters, consonants, offsets, vfs, ends) = CalcPositionsAndRatiosMs(phones);
            for (int i = 0; i < n; i++) {
                ProcessSinglePhoneme(phrase, phones[i], melExc, starts[i], preutters[i], consonants[i], offsets[i], vfs[i], ends[i]);
            }
        }

        /// <summary>fragment_mel.py calc_positions_and_ratios_ms（单位：ms / 采样）。</summary>
        public static (double[] starts, double[] preutters, double[] consonants, double[] offsets,
                       double[] vfs, double[] ends) CalcPositionsAndRatiosMs(System.Collections.Generic.List<PhoneInfo> phones) {
            double spm = 44100 / 1000.0;
            int n = phones.Count;
            var startsMs = new double[n];
            var starts = new double[n];
            var preutters = new double[n];
            var consonants = new double[n];
            var offsets = new double[n];
            var vfs = new double[n];
            var ends = new double[n];

            for (int i = 0; i < n; i++) {
                var info = phones[i];
                double p0 = info.P0.X, p1 = info.P1.X, p4 = info.P4.X;
                double ov = p1 - p0;
                double s;
                if (i == 0) {
                    s = 0.0;
                } else {
                    var prev = phones[i - 1];
                    double prevLen = prev.P4.X - prev.P0.X;
                    s = startsMs[i - 1] + prevLen - ov;
                }
                double preutter = s - p0;
                double consonantMs = info.Oto.Consonant;
                double preutterMs = info.Oto.Preutter;
                double vel = info.RequireFlag("vel");
                double vf = Math.Pow(2.0, (100.0 - vel) / 100.0);
                double consonant = preutter + (consonantMs - preutterMs) * vf;
                double offset = consonant - consonantMs * vf;
                double endPos = s + (p4 - p0);

                startsMs[i] = s;
                starts[i] = s * spm;
                preutters[i] = preutter * spm;
                consonants[i] = consonant * spm;
                offsets[i] = offset * spm;
                vfs[i] = vf;
                ends[i] = endPos * spm;
            }
            return (starts, preutters, consonants, offsets, vfs, ends);
        }

        /// <summary>fragment_mel.py _process_single_phoneme。</summary>
        public static void ProcessSinglePhoneme(PhraseData phrase, PhoneInfo info, MelExtractor melExc,
                double starts, double preutter, double consonant, double offset, double vf, double end) {
            double spm = 44100 / 1000.0;
            var audio = AudioReader.Read(info.Oto.AudioFilePath, 44100);
            double totalLen = audio.Length;

            double otoOffset = info.Oto.Offset * spm;
            double otoConsonant = (info.Oto.Consonant + info.Oto.Offset) * spm;
            double otoEnd = info.Oto.Cutoff > 0
                ? totalLen - info.Oto.Cutoff * spm
                : otoOffset + Math.Abs(info.Oto.Cutoff * spm);
            double otoPreutter = (info.Oto.Preutter + info.Oto.Offset) * spm;

            int strt = (int)info.Flag("strt", 0);
            double sf = otoEnd - otoConsonant > 0 ? (end - consonant) / (otoEnd - otoConsonant) : 1.0;

            int melOffset = (int)Math.Floor((offset - HopLength / 2.0) / HopLength);
            int melConsonant = (int)Math.Floor((consonant - HopLength / 2.0) / HopLength);
            int melEnd = (int)Math.Floor((end - HopLength / 2.0) / HopLength + 1);
            int frame = melEnd - melOffset + 1;
            int conFrames = melEnd - melConsonant + 1;

            // Python 浮点取模: 结果恒为非负（C# % 对负数返回负值，需修正）
            double pyMod = (offset - HopLength / 2.0) % HopLength;
            if (pyMod < 0) pyMod += HopLength;
            double hStart = offset - pyMod;
            var hPoints = new double[frame];
            for (int i = 0; i < frame; i++) hPoints[i] = i * (double)HopLength + hStart;

            // fn(x): 辅音段速度拉伸，元音段按 strt 线性或三角循环
            var otoHPoints = new double[frame];
            for (int i = 0; i < frame; i++) {
                double x = hPoints[i];
                if (x < consonant) {
                    otoHPoints[i] = otoConsonant - (consonant - x) / vf;
                } else if (strt == 1) {
                    double period = 2 * (otoEnd - otoConsonant);
                    otoHPoints[i] = otoEnd - Math.Abs((x - consonant) % period - (otoEnd - otoConsonant));
                } else {
                    otoHPoints[i] = otoConsonant + (x - consonant) / sf;
                }
            }

            var mel = melExc.Extract(audio, otoHPoints);
            int T = frame;
            // dynamic_range_compression
            for (int i = 0; i < mel.Length; i++) mel[i] = Math.Log(Math.Max(mel[i], 1e-5));

            // strt=1: 抹平循环段能量起伏（前 T-conFrames 帧为元音）
            if (strt == 1 && conFrames < T) {
                int vowelLen = T - conFrames;
                var frameEnergy = new double[vowelLen];
                for (int t = 0; t < vowelLen; t++) {
                    double sum = 0;
                    for (int m = 0; m < 128; m++) sum += Math.Exp(mel[m * T + t]);
                    frameEnergy[t] = Math.Max(sum / 128, 1e-12);
                }
                double e0 = frameEnergy[0], e1 = frameEnergy[vowelLen - 1];
                for (int t = 0; t < vowelLen; t++) {
                    double target = vowelLen > 1 ? e0 + (e1 - e0) * t / (vowelLen - 1) : e0;
                    double logT = Math.Log(target);
                    for (int m = 0; m < 128; m++) {
                        mel[m * T + t] += -Math.Log(frameEnergy[t]) + logT;
                    }
                }
            }

            // P 参数：音量均衡
            double pFlag = info.Flag("P", 0);
            if (pFlag > 0 && T > 0) {
                double targetRms = 0.5;
                double sumSq = 0;
                for (int i = 0; i < mel.Length; i++) {
                    double v = Math.Exp(mel[i]);
                    sumSq += v * v;
                }
                double curRms = Math.Sqrt(sumSq / mel.Length);
                if (curRms > 1e-12) {
                    double blend = pFlag / 100.0;
                    double targetRmsActual = curRms * (1 - blend) + targetRms * blend;
                    double logScale = Math.Log(targetRmsActual / curRms);
                    for (int i = 0; i < mel.Length; i++) mel[i] += logScale;
                }
            }

            info.Mel = mel;
            info.MelFrames = T;
            info.AudioSeg = audio;
            info.MelOffset = melOffset;
            info.MelEnd = melEnd;
            info.Preutter = preutter;
            info.HPoints = hPoints;
        }

        /// <summary>fragment_mel.py adjust_volume_by_phtp（与 Fragment 相同逻辑，但 ms_per_frame=hop512）。</summary>
        public static void AdjustVolumeByPhtp(PhraseData phrase) {
            Fragment.AdjustVolumeByPhtp(phrase, MsPerFrame);
        }

        /// <summary>fragment_mel.py apply_dynamic_gen_to_mels（与 Fragment 相同逻辑）。</summary>
        public static void ApplyDynamicGenToMels(PhraseData phrase) {
            Fragment.ApplyDynamicGenToMels(phrase);
        }
    }
}
