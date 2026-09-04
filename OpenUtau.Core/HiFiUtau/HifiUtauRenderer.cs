using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using K4os.Hash.xxHash;
using NAudio.Wave;
using Newtonsoft.Json.Linq;
using OpenUtau.Core;
using OpenUtau.Core.Format;
using OpenUtau.Core.HiFiUtau.Engine.Pipeline;
using OpenUtau.Core.Render;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HiFiUtau {
    /// <summary>
    /// HiFiUTAU 渲染器 — 内嵌引擎（进程内 ONNX 推理）+ 三级缓存（hifigan / hnsep / final）。
    ///
    /// 管线三段（对应原 HTTP 端点 /syn_mel、/syn_hnsep、/syn_post，现为进程内调用）：
    ///   1. mel 拼接 + 变调(genc) + HiFi-GAN → 缓存 hifiutau/hifigan/{hifiganHash}.wav
    ///   2. HN-SEP 气声/谐波分离            → 缓存 hifiutau/hnsep/{hifiganHash}.{harmonic,noise}.wav
    ///   3. 参数应用                        → 缓存 hifiutau/final/{finalHash}.wav
    ///
    /// 缓存命中规则：
    ///   - final 命中 → 直接使用（最快）
    ///   - 修改旋律/歌词/变调 → 只重算 hifigan（hnsep/final 哈希随 hifigan 变化）
    ///   - 修改参数（但 hifigan 不变）→ hnsep 缓存直接命中，只重算 post
    ///   - 所有 HN-SEP 相关参数均为默认值 → 完全跳过分离
    ///
    /// 本渲染器不回退到 HTTP——模型缺失或内嵌引擎被禁用时抛出明确错误；
    /// HTTP 仅属于 Custom Server 渲染器。
    /// </summary>
    public class HifiUtauRenderer : IRenderer {
        // 基于 phrase.hash 的互斥锁，防止相同内容并发重复提交
        private static readonly ConcurrentDictionary<ulong, SemaphoreSlim> _hashLocks =
            new ConcurrentDictionary<ulong, SemaphoreSlim>();

        public HifiUtauRenderer() {
        }

        public USingerType SingerType => USingerType.Classic;

        public bool SupportsRenderPitch => false;

        public bool SupportsExpression(UExpressionDescriptor descriptor) {
            return true;
        }

        public RenderResult Layout(RenderPhrase phrase) {
            return new RenderResult() {
                leadingMs = phrase.leadingMs,
                positionMs = phrase.positionMs,
                estimatedLengthMs = phrase.durationMs + phrase.leadingMs,
            };
        }

        public Task<RenderResult> Render(RenderPhrase phrase, Progress progress, int trackNo,
            CancellationTokenSource cancellation, bool isPreRender = false, RenderPhraseEvents? renderEvents = null) {
            return RenderImpl(phrase, progress, trackNo, cancellation, isPreRender);
        }

        internal async Task<RenderResult> RenderImpl(RenderPhrase phrase, Progress progress, int trackNo,
            CancellationTokenSource cancellation, bool isPreRender) {
            string progressInfo =
                $"Track {trackNo + 1}: HifiUtau \"{string.Join(" ", phrase.phones.Select(p => p.phoneme))}\"";
            var result = Layout(phrase);

            // ── 计算缓存哈希 ──
            ulong hifiganHash = ComputeHifiganHash(phrase);
            ulong finalHash = ComputeFinalHash(phrase, hifiganHash);
            bool needsHnsep = NeedsHnsep(phrase);

            var cacheRoot = Path.Join(PathManager.Inst.CachePath, "hifiutau");
            var hifiganDir = Path.Join(cacheRoot, "hifigan");
            var hnsepDir = Path.Join(cacheRoot, "hnsep");
            var finalDir = Path.Join(cacheRoot, "final");
            Directory.CreateDirectory(hifiganDir);
            Directory.CreateDirectory(hnsepDir);
            Directory.CreateDirectory(finalDir);

            string hifiganPath = Path.Join(hifiganDir, $"{hifiganHash:x16}.wav");
            string harmonicPath = Path.Join(hnsepDir, $"{hifiganHash:x16}.harmonic.wav");
            string noisePath = Path.Join(hnsepDir, $"{hifiganHash:x16}.noise.wav");
            string finalPath = Path.Join(finalDir, $"{finalHash:x16}.wav");
            phrase.AddCacheFile(finalPath);

            // 快路径：final 缓存命中（无锁）
            if (File.Exists(finalPath)) {
                result.samples = LoadSamples(finalPath);
                if (result.samples != null) {
                    Renderers.ApplyDynamics(phrase, result);
                }
                progress.Complete(phrase.phones.Length, progressInfo);
                return result;
            }

            // 基于 hash 的互斥锁 + double-check
            var hashLock = GetOrCreateHashLock(phrase.hash);
            await hashLock.WaitAsync(cancellation.Token).ConfigureAwait(false);
            try {
                if (File.Exists(finalPath)) {
                    result.samples = LoadSamples(finalPath);
                    if (result.samples != null) {
                        Renderers.ApplyDynamics(phrase, result);
                    }
                    progress.Complete(phrase.phones.Length, progressInfo);
                    return result;
                }

                // 基础音素 JSON（一次构建，各段复用）
                var baseJson = HifiUtauPhraseJson.Build(phrase);

                // ── 阶段1: mel 拼接 + 变调 + HiFi-GAN ──
                if (!File.Exists(hifiganPath)) {
                    var payload = (JObject)baseJson.DeepClone();
                    payload["out_wav"] = hifiganPath;
                    byte[] wav;
                    lock (HiFiUtauModelStore.Inst.EngineLock) {
                        // 引擎引用只在 EngineLock 内获取与使用：重载同样持锁，
                        // 保证"会话释放"与"会话使用"永不交错
                        wav = RequireEngine().SynthesizeMel(payload);
                    }
                    File.WriteAllBytes(hifiganPath, wav);
                }

                // ── 阶段2: HN-SEP 气声/谐波分离（仅当需要时） ──
                if (needsHnsep && (!File.Exists(harmonicPath) || !File.Exists(noisePath))) {
                    byte[] harmonic, noise;
                    lock (HiFiUtauModelStore.Inst.EngineLock) {
                        (harmonic, noise) = RequireEngine().SynthesizeHnsep(File.ReadAllBytes(hifiganPath));
                    }
                    File.WriteAllBytes(harmonicPath, harmonic);
                    File.WriteAllBytes(noisePath, noise);
                }

                // ── 阶段3: 参数应用 ──
                var postPayload = (JObject)baseJson.DeepClone();
                byte[]? wavBytes = null, harmonicBytes = null, noiseBytes = null;
                if (needsHnsep) {
                    harmonicBytes = File.ReadAllBytes(harmonicPath);
                    noiseBytes = File.ReadAllBytes(noisePath);
                } else {
                    wavBytes = File.ReadAllBytes(hifiganPath);
                }
                byte[] finalWav;
                lock (HiFiUtauModelStore.Inst.EngineLock) {
                    finalWav = RequireEngine().SynthesizePost(postPayload, wavBytes, harmonicBytes, noiseBytes);
                }
                File.WriteAllBytes(finalPath, finalWav);
            } finally {
                hashLock.Release();
            }

            if (!File.Exists(finalPath)) {
                throw new IOException("final 缓存文件未生成");
            }
            result.samples = LoadSamples(finalPath);
            if (result.samples != null) {
                Renderers.ApplyDynamics(phrase, result);
            }
            progress.Complete(phrase.phones.Length, progressInfo);
            return result;
        }

        /// <summary>
        /// 解析可用的内嵌引擎；不可用时抛出带可翻译说明的错误（由渲染管线弹出错误框）。
        /// 每个阶段调用一次，使设置变更/重载能在短语渲染间隙生效。
        /// </summary>
        private static SynthesisEngine RequireEngine() {
            if (!Preferences.Default.HifiUtauEmbedded) {
                throw new MessageCustomizableException(
                    "HiFiUTAU embedded engine is disabled in preferences (Preferences > Rendering > HiFiUTAU).",
                    "<translate:errors.hifiutau.disabled>",
                    null,
                    false);
            }
            var engine = HiFiUtauModelStore.Inst.GetEngine();
            if (engine == null) {
                var store = HiFiUtauModelStore.Inst;
                string detail = string.Join(" | ",
                    new[] { store.SplicerError, store.HnsepError }.Where(s => !string.IsNullOrEmpty(s)));
                throw new MessageCustomizableException(
                    $"HiFiUTAU embedded engine models are missing or failed to load. {detail}",
                    "<translate:errors.hifiutau.nomodels>",
                    null,
                    false);
            }
            return engine;
        }

        // ════════════════════════════════════════════════════════════════
        //  缓存哈希
        // ════════════════════════════════════════════════════════════════

        /// <summary>
        /// hifigan 缓存哈希：所有影响 HiFi-GAN 之前输入的字段
        /// （音素/oto/vel/vol/phtp/envelope/splc 已含在 phone.hash；另加 F0 与 genc）。
        /// </summary>
        private static ulong ComputeHifiganHash(RenderPhrase phrase) {
            using (var stream = new MemoryStream()) {
                using (var writer = new BinaryWriter(stream)) {
                    writer.Write(phrase.singer?.Id ?? string.Empty);
                    writer.Write(phrase.renderer?.ToString() ?? string.Empty);
                    writer.Write(phrase.timeAxis.Timestamp);
                    foreach (var phone in phrase.phones) {
                        writer.Write(phone.hash);
                    }
                    WriteFloatArray(writer, phrase.pitches); // F0（含 pitd 偏差）
                    WriteFloatArray(writer, phrase.gender);  // genc
                    return XXH64.DigestOf(stream.ToArray());
                }
            }
        }

        /// <summary>
        /// final 缓存哈希：hifiganHash + 所有影响参数应用阶段的曲线。
        /// dyn 除外（仍由 C# 端 ApplyDynamics 施加，不入缓存键）。
        /// </summary>
        private static ulong ComputeFinalHash(RenderPhrase phrase, ulong hifiganHash) {
            using (var stream = new MemoryStream()) {
                using (var writer = new BinaryWriter(stream)) {
                    writer.Write(hifiganHash);
                    WriteFloatArray(writer, phrase.breathiness);  // brec
                    WriteFloatArray(writer, phrase.tension);      // tenc
                    WriteFloatArray(writer, phrase.voicing);      // voic
                    WriteFloatArray(writer, phrase.breathLow);    // brel
                    WriteFloatArray(writer, phrase.breathHigh);   // breh
                    WriteFloatArray(writer, phrase.warmth);       // warm
                    WriteFloatArray(writer, phrase.hcmp);         // hcmp
                    WriteFloatArray(writer, phrase.lowcut);       // lowc
                    // growl (gwl) 位于自定义曲线中
                    var growl = phrase.curves?.FirstOrDefault(c => c.Item1 == Format.Ustx.GWL)?.Item2;
                    WriteFloatArray(writer, growl);
                    return XXH64.DigestOf(stream.ToArray());
                }
            }
        }

        /// <summary>
        /// 是否需要 HN-SEP 分离：与引擎侧判定一致。
        /// breath/tension/brel/breh/warm/hcmp 任一处偏离默认 0，或 voicing 偏离默认 100。
        /// </summary>
        private static bool NeedsHnsep(RenderPhrase phrase) {
            return AnyNotClose(phrase.breathiness, 0, 0.5)
                || AnyNotClose(phrase.tension, 0, 0.5)
                || !AllClose(phrase.voicing, 100, 0.05)
                || AnyNotClose(phrase.breathLow, 0, 0.5)
                || AnyNotClose(phrase.breathHigh, 0, 0.5)
                || AnyNotClose(phrase.warmth, 0, 0.5)
                || AnyNotClose(phrase.hcmp, 0, 0.5);
        }

        private static bool AnyNotClose(float[] arr, float value, double atol) {
            if (arr == null || arr.Length == 0) {
                return false;
            }
            foreach (var v in arr) {
                if (Math.Abs(v - value) > atol) {
                    return true;
                }
            }
            return false;
        }

        private static bool AllClose(float[] arr, float value, double rtol) {
            if (arr == null || arr.Length == 0) {
                return true;
            }
            foreach (var v in arr) {
                if (Math.Abs(v - value) > rtol * Math.Abs(value) + 1e-8) {
                    return false;
                }
            }
            return true;
        }

        private static void WriteFloatArray(BinaryWriter writer, float[] array) {
            if (array == null) {
                writer.Write("null");
                return;
            }
            foreach (var v in array) {
                writer.Write(v);
            }
        }

        // ════════════════════════════════════════════════════════════════
        //  其它
        // ════════════════════════════════════════════════════════════════

        private static float[] LoadSamples(string path) {
            using (var waveStream = new WaveFileReader(path)) {
                return Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0));
            }
        }

        private static SemaphoreSlim GetOrCreateHashLock(ulong hash) {
            var newLock = new SemaphoreSlim(1, 1);
            var hashLock = _hashLocks.GetOrAdd(hash, newLock);
            if (hashLock != newLock) {
                newLock.Dispose();
            }
            return hashLock;
        }

        public RenderPitchResult LoadRenderedPitch(RenderPhrase phrase) {
            return null!;
        }

        public List<RenderRealCurveResult> LoadRenderedRealCurves(RenderPhrase phrase) {
            return new List<RenderRealCurveResult>(0);
        }

        public UExpressionDescriptor[] GetSuggestedExpressions(USinger singer, URenderSettings renderSettings) {
            return new UExpressionDescriptor[] { };
        }

        public override string ToString() => Renderers.HIFIUTAU;
    }
}
