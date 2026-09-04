using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using OpenUtau.Core.Render;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;

namespace OpenUtau.Core.CustomRender {
    /// <summary>
    /// 将 RenderPhrase 转换为 Custom Server HTTP 端点所需的音素 JSON。
    /// 独立于 HiFiUTAU，可单独维护。
    /// </summary>
    public static class CustomPhraseJson {
        /// <summary>
        /// 构建完整音素 JSON（hop_size/frame_ms/phoneme_list/Dynamic_parameter）。
        /// </summary>
        public static JObject Build(RenderPhrase phrase) {
            const int hopSize = 256;
            const int sampleRate = 44100;
            double frameMs = 1000.0 * hopSize / sampleRate;

            int totalFrames = (int)Math.Ceiling((phrase.durationMs + phrase.leadingMs) / frameMs);

            var curves = CustomF0Utils.SampleAllCurves(phrase, frameMs, totalFrames);

            var dynamicParam = new Dictionary<string, object?>();

            var knownCurves = new[] { "pitd", "genc", "brec", "tenc", "voic", "lowc", "brel", "breh" };
            foreach (var abbr in knownCurves) {
                if (curves.TryGetValue(abbr, out var data)) {
                    dynamicParam[abbr] = data;
                }
            }

            foreach (var kvp in curves) {
                if (Array.IndexOf(knownCurves, kvp.Key) >= 0) {
                    continue;
                }
                dynamicParam[kvp.Key] = kvp.Value;
            }

            var perPhonemeCurves = new Dictionary<string, Dictionary<string, double[]>>();
            for (int i = 0; i < phrase.phones.Length; i++) {
                var phone = phrase.phones[i];
                var curvesForPhone = CustomF0Utils.SampleCurvesForPhoneme(
                    phrase, phone, i, frameMs, totalFrames);
                perPhonemeCurves[(i + 1).ToString()] = curvesForPhone;
            }

            var phonemeList = BuildPhonemeList(phrase, perPhonemeCurves);

            var jsonData = new {
                hop_size = hopSize,
                sample_rate = sampleRate,
                frame_ms = frameMs,
                out_wav = Path.Join(PathManager.Inst.CachePath, $"custom-{phrase.hash:x16}.wav"),
                wav_dur = phrase.durationMs,
                phoneme_list = phonemeList,
                Dynamic_parameter = dynamicParam,
            };

            return JObject.FromObject(jsonData);
        }

        private static Dictionary<string, object> BuildPhonemeList(
            RenderPhrase phrase,
            Dictionary<string, Dictionary<string, double[]>> perPhonemeCurves) {
            var phonemeList = new Dictionary<string, object>();

            for (int i = 0; i < phrase.phones.Length; i++) {
                var phone = phrase.phones[i];
                var envelope = phone.envelope;
                var key = (i + 1).ToString();

                var noteFlags = new Dictionary<string, object> {
                    { "vel", phone.velocity * 100.0 },
                    { "vol", phone.volume * 100.0 },
                    { "mod", phone.modulation * 100.0 },
                    { "shft", phone.toneShift },
                    { "phtp", phone.phonemeType },
                    { "strt", phone.stretchMode },
                    { "splc", phone.spliceMode },
                    { "stms", phone.stretchMs },
                    { "S", phone.stretchMs }
                };
                foreach (var flag in phone.flags) {
                    noteFlags[flag.Item1] = flag.Item2 ?? 0;
                }

                float p3X, p3Y, p4X, p4Y;

                if (i < phrase.phones.Length - 1) {
                    var nextPhone = phrase.phones[i + 1];
                    var nextEnvelope = nextPhone.envelope;

                    var nextPreutter = -nextEnvelope[0].X;
                    var nextOverlap = nextEnvelope[1].X - nextEnvelope[0].X;
                    var tailIntrude = Math.Max(nextPreutter, nextPreutter - nextOverlap);
                    var tailOverlap = Math.Max(nextOverlap, 0);

                    p3X = (float)(phone.durationMs - tailIntrude);
                    p4X = (float)(p3X + tailOverlap);
                    p3Y = envelope[3].Y;
                    p4Y = envelope[4].Y;
                } else {
                    p3X = envelope[3].X;
                    p3Y = envelope[3].Y;
                    p4X = envelope[4].X;
                    p4Y = envelope[4].Y;
                }

                Dictionary<string, double[]>? phonemeDynamicParams = null;
                if (perPhonemeCurves.TryGetValue(key, out var curves) && curves != null && curves.Count > 0) {
                    phonemeDynamicParams = curves;
                }

                var phonemeData = new {
                    phoneme_name = phone.phoneme,
                    note_pitch = MusicMath.GetToneName(phone.tone),
                    dur = envelope[4].X,
                    Note_flags = noteFlags,
                    phoneme_oto = new {
                        audio_file_path = phone.oto?.File ?? "",
                        Offset = phone.oto?.Offset ?? 0.0,
                        Consonant = phone.oto?.Consonant ?? 0.0,
                        Cutoff = phone.oto?.Cutoff ?? 0.0,
                        Preutter = phone.oto?.Preutter ?? 0.0,
                        Overlap = phone.oto?.Overlap ?? 0.0,
                    },
                    envelope = new {
                        p0 = new { x = envelope[0].X, y = envelope[0].Y },
                        p1 = new { x = envelope[1].X, y = envelope[1].Y },
                        p2 = new { x = envelope[2].X, y = envelope[2].Y },
                        p3 = new { x = p3X, y = p3Y },
                        p4 = new { x = p4X, y = p4Y }
                    },
                    Dynamic_parameter = phonemeDynamicParams
                };
                phonemeList[key] = phonemeData;
            }

            return phonemeList;
        }
    }
}
