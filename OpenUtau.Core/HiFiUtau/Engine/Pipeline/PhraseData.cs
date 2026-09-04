using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// hifiutau-engine 合成请求 JSON 的强类型解析。
    /// JSON 契约与 HifiUtauPhraseJson.Build() 的输出一致（即 Python 引擎的输入）。
    /// </summary>
    public sealed class PhraseData {
        public int HopSize { get; init; }                 // json['hop_size']，Python: Dynamic_hop
        public double WavDurMs { get; init; }             // json['wav_dur']
        public int SampleRateConst => 44100;              // frag.sample_rate（Python 固定 44100）
        public List<PhoneInfo> Phones { get; init; } = new();
        public Dictionary<string, double[]> Dynamic { get; init; } = new();  // Dynamic_parameter

        public static PhraseData Parse(JObject json) {
            var data = new PhraseData {
                HopSize = json.Value<int?>("hop_size") ?? 256,
                WavDurMs = json.Value<double?>("wav_dur") ?? 0,
            };
            if (json["Dynamic_parameter"] is JObject dp) {
                foreach (var kv in dp) {
                    if (kv.Value is JArray arr) {
                        var curve = new double[arr.Count];
                        for (int i = 0; i < arr.Count; i++) curve[i] = arr[i].Value<double>();
                        data.Dynamic[kv.Key] = curve;
                    }
                }
            }
            if (json["phoneme_list"] is JObject list) {
                foreach (var kv in list) {
                    if (kv.Value is JObject p) data.Phones.Add(PhoneInfo.Parse(p));
                }
            }
            return data;
        }

        /// <summary>fragment.py _get_param：取第一个存在的键，否则空数组。</summary>
        public double[] GetParam(params string[] keys) {
            foreach (string k in keys) {
                if (Dynamic.TryGetValue(k, out var v)) return v;
            }
            return Array.Empty<double>();
        }
    }

    public sealed class PhoneInfo {
        public string Name = "";
        public Dictionary<string, double> Flags = new();   // Note_flags
        public OtoEntry Oto = new();
        public EnvPoint P0 = new(), P1 = new(), P2 = new(), P3 = new(), P4 = new();
        public Dictionary<string, double[]> PerPhoneDP = new();  // 音素级 Dynamic_parameter

        // ── 管线中间产物 ──
        public double[] Mel = Array.Empty<double>();      // (128, T) band-major
        public int MelFrames;                             // Mel 的帧数
        public float[] AudioSeg = Array.Empty<float>();
        public int ConsonantFrames;
        public double StretchFactor = 1.0;
        // FragmentMel 专用
        public double[] HPoints = Array.Empty<double>();
        public int MelOffset, MelEnd;
        public double Preutter;
        // BaseSplicer 专用
        public int ModelStartFrame, ModelEndFrame, ModelFrames;

        public double Flag(string key, double fallback) => Flags.TryGetValue(key, out var v) ? v : fallback;
        public double RequireFlag(string key) => Flags.TryGetValue(key, out var v) ? v : throw new KeyNotFoundException($"Note_flags 缺少 {key}");

        public double[] GetPhoneDP(string key)
            => PerPhoneDP.TryGetValue(key, out var v) && v.Length > 0 ? v : Array.Empty<double>();

        public static PhoneInfo Parse(JObject p) {
            var info = new PhoneInfo { Name = p.Value<string>("phoneme_name") ?? "" };
            if (p["Note_flags"] is JObject flags) {
                foreach (var kv in flags) {
                    info.Flags[kv.Key] = kv.Value?.Type == JTokenType.Array
                        ? 0 : Convert.ToDouble(kv.Value?.Value<double?>() ?? kv.Value?.Value<long?>() ?? 0);
                }
            }
            if (p["phoneme_oto"] is JObject oto) {
                info.Oto = new OtoEntry {
                    AudioFilePath = oto.Value<string>("audio_file_path") ?? "",
                    Offset = oto.Value<double?>("Offset") ?? 0,
                    Consonant = oto.Value<double?>("Consonant") ?? 0,
                    Cutoff = oto.Value<double?>("Cutoff") ?? 0,
                    Preutter = oto.Value<double?>("Preutter") ?? 0,
                };
            }
            if (p["envelope"] is JObject env) {
                info.P0 = ParsePoint(env["p0"]);
                info.P1 = ParsePoint(env["p1"]);
                info.P2 = ParsePoint(env["p2"]);
                info.P3 = ParsePoint(env["p3"]);
                info.P4 = ParsePoint(env["p4"]);
            }
            if (p["Dynamic_parameter"] is JObject dp) {
                foreach (var kv in dp) {
                    if (kv.Value is JArray arr) {
                        var curve = new double[arr.Count];
                        for (int i = 0; i < arr.Count; i++) curve[i] = arr[i].Value<double>();
                        info.PerPhoneDP[kv.Key] = curve;
                    }
                }
            }
            return info;
        }

        private static EnvPoint ParsePoint(JToken? t) {
            var p = t as JObject;
            return new EnvPoint {
                X = p?.Value<double?>("x") ?? 0,
                Y = p?.Value<double?>("y") ?? 0,
            };
        }
    }

    public sealed class OtoEntry {
        public string AudioFilePath = "";
        public double Offset, Consonant, Cutoff, Preutter;
    }

    public sealed class EnvPoint {
        public double X, Y;
    }
}
