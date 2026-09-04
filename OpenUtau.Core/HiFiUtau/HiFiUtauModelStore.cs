using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Core.HiFiUtau.Engine.Pipeline;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HiFiUtau {
    /// <summary>
    /// 内嵌 HiFiUTAU 引擎的模型定位与加载（进程内推理，无需 Python/HTTP）。
    ///
    /// 两款模型独立加载：
    ///   - 声码器：只接受 OpenUtau vocoder oudep 布局（vocoder.yaml + ONNX），
    ///     默认 id <c>pc_nsf_hifigan_44.1k_hop512_128bin_2025.02</c>。不加载 part1/part2，也不加载 PyTorch ckpt。
    ///   - 分离模型：只接受 HN-SEP oudep 布局（oudep.yaml + ONNX），
    ///     默认 id <c>hnsep_VR_44.1k_hop512_2024.05</c>。
    /// 探测：偏好设置路径（须为 oudep 解包目录）→ Dependencies 已安装 oudep。
    /// ONNX 会话经 Util.Onnx 创建（复用"渲染"设置中的 CPU/DirectML 运行时选择），
    /// 并应用偏好中的推理线程数。
    /// 模型缺失时不做任何回退——由渲染器抛出明确错误；HTTP 仅属于 Custom Server 渲染器。
    /// </summary>
    public sealed class HiFiUtauModelStore {
        private static readonly Lazy<HiFiUtauModelStore> lazy = new(() => new HiFiUtauModelStore());
        public static HiFiUtauModelStore Inst => lazy.Value;

        private readonly object loadLock = new();          // 保护懒加载与重载
        /// <summary>引擎推理串行化锁（重载时也持有它，保证在途渲染完成后再释放旧会话）。</summary>
        public readonly object EngineLock = new object();

        private HiddenSplicer? splicer;
        private Hnsep? hnsep;
        private SynthesisEngine? engine;

        public string? SplicerDir { get; private set; }
        public string? HnsepDir { get; private set; }
        public bool SplicerReady => splicer != null;
        public bool HnsepReady => hnsep != null;
        public bool EngineReady => engine != null;
        public string? SplicerError { get; private set; }
        public string? HnsepError { get; private set; }

        /// <summary>svs-index 上的 PC-NSF-HiFiGAN 2025.02 vocoder oudep id。</summary>
        public const string PcNsfHifiGanOudepId = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02";

        /// <summary>WORLDLINE-R2 使用的旧目录名，部分安装仍是这个 id。</summary>
        public const string PcNsfHifiGanLegacyOudepId = "pc-nsf-hifigan";

        /// <summary>HN-SEP VR 2024.05 oudep id（yxlllc/vocal-remover hnsep_240512）。</summary>
        public const string HnsepOudepId = "hnsep_VR_44.1k_hop512_2024.05";

        private HiFiUtauModelStore() { }

        // ── 目录探测 ──

        public static bool IsValidVocoderOudepDir(string dir) {
            if (!File.Exists(Path.Combine(dir, "vocoder.yaml"))) {
                return false;
            }
            try {
                var cfg = ReadVocoderYaml(dir);
                return File.Exists(Path.Combine(dir, cfg.Model));
            } catch {
                return false;
            }
        }

        public static bool LooksLikePytorchCkptDir(string dir)
            => File.Exists(Path.Combine(dir, "model.ckpt"))
            && (File.Exists(Path.Combine(dir, "config.json"))
                || File.Exists(Path.Combine(dir, "config.yaml")));

        static FusedVocoderYaml ReadVocoderYaml(string dir) {
            var cfg = Yaml.DefaultDeserializer.Deserialize<FusedVocoderYaml>(
                File.ReadAllText(Path.Combine(dir, "vocoder.yaml")));
            if (string.IsNullOrEmpty(cfg.Model)) {
                cfg.Model = "model.onnx";
            }
            return cfg;
        }

        public static bool IsValidHnsepOudepDir(string dir) {
            if (!File.Exists(Path.Combine(dir, "oudep.yaml"))) {
                return false;
            }
            return Directory.GetFiles(dir, "*.onnx").Length > 0;
        }

        /// <summary>解析声码器目录（偏好设置覆盖 → 已安装 vocoder oudep）。</summary>
        public string? FindSplicerDir() {
            string? custom = Preferences.Default.HifiUtauSplicerPath;
            if (!string.IsNullOrWhiteSpace(custom)) {
                return IsValidVocoderOudepDir(custom) ? custom : null;
            }
            foreach (string id in new[] { PcNsfHifiGanOudepId, PcNsfHifiGanLegacyOudepId }) {
                string? installed = PackageManager.Inst.GetInstalledPath(id);
                if (!string.IsNullOrEmpty(installed) && IsValidVocoderOudepDir(installed)) {
                    return installed;
                }
            }
            return null;
        }

        /// <summary>解析分离模型目录（偏好设置覆盖 → 已安装 HN-SEP oudep）。</summary>
        public string? FindHnsepDir() {
            string? custom = Preferences.Default.HifiUtauHnsepPath;
            if (!string.IsNullOrWhiteSpace(custom)) {
                return IsValidHnsepOudepDir(custom) ? custom : null;
            }
            string? installed = PackageManager.Inst.GetInstalledPath(HnsepOudepId);
            if (!string.IsNullOrEmpty(installed) && IsValidHnsepOudepDir(installed)) {
                return installed;
            }
            return null;
        }

        // ── 懒加载 ──

        /// <summary>
        /// 获取内嵌引擎；任一模型缺失或加载失败返回 null（渲染器应抛出明确错误）。
        /// 线程安全；模型仅加载一次。
        /// </summary>
        public SynthesisEngine? GetEngine() {
            if (engine != null) return engine;
            lock (loadLock) {
                if (engine != null) return engine;
                var s = GetSplicer();
                if (s == null) return null;
                var h = GetHnsep();
                if (h == null) return null;
                engine = new SynthesisEngine(s, h, new MelExtractor());
                Log.Information("HiFiUTAU 内嵌引擎就绪");
                return engine;
            }
        }

        private HiddenSplicer? GetSplicer() {
            if (splicer != null) return splicer;
            lock (loadLock) {
                if (splicer != null) return splicer;
                try {
                    string? dir = FindSplicerDir();
                    if (dir == null) {
                        SplicerDir = null;
                        string custom = Preferences.Default.HifiUtauSplicerPath;
                        if (!string.IsNullOrWhiteSpace(custom) && LooksLikePytorchCkptDir(custom)) {
                            SplicerError = "该目录是 PC-NSF-HiFiGAN 的 PyTorch ckpt 包（model.ckpt）。进程内 HiFiUTAU 只加载 OpenUtau vocoder oudep（vocoder.yaml + ONNX）。请用包管理器安装 pc_nsf_hifigan_44.1k_hop512_128bin_2025.02。";
                        } else {
                            SplicerError = "未找到声码器。请在包管理器安装 pc_nsf_hifigan_44.1k_hop512_128bin_2025.02。";
                        }
                        Log.Warning("HiFiUTAU 内嵌引擎：{err}", SplicerError);
                        return null;
                    }
                    SplicerDir = dir;
                    Log.Information("HiFiUTAU 内嵌引擎：加载声码器 {dir}", dir);
                    splicer = LoadSplicer(dir);
                    SplicerError = null;
                } catch (Exception e) {
                    SplicerError = e.Message;
                    Log.Error(e, "HiFiUTAU 内嵌引擎：拼接模型加载失败");
                }
                return splicer;
            }
        }

        private Hnsep? GetHnsep() {
            if (hnsep != null) return hnsep;
            lock (loadLock) {
                if (hnsep != null) return hnsep;
                try {
                    string? dir = FindHnsepDir();
                    if (dir == null) {
                        HnsepDir = null;
                        HnsepError = "未找到 HN-SEP。请在包管理器安装 hnsep_VR_44.1k_hop512_2024.05。";
                        Log.Warning("HiFiUTAU 内嵌引擎：{err}", HnsepError);
                        return null;
                    }
                    HnsepDir = dir;
                    string modelPath = PickHnsepModel(dir);
                    Log.Information("HiFiUTAU 内嵌引擎：加载分离模型 {model}", modelPath);
                    hnsep = new Hnsep(modelPath, CreateSession);
                    HnsepError = null;
                } catch (Exception e) {
                    HnsepError = e.Message;
                    Log.Error(e, "HiFiUTAU 内嵌引擎：分离模型加载失败");
                }
                return hnsep;
            }
        }

        HiddenSplicer LoadSplicer(string dir) {
            var cfg = ReadVocoderYaml(dir);
            string modelPath = Path.Combine(dir, cfg.Model);
            var session = CreateSession(modelPath);
            return new HiddenSplicer(session, cfg.HopSize, cfg.SampleRate, cfg.NumMelBins);
        }

        private InferenceSession CreateSession(string path) {
            return Onnx.getInferenceSession(path, OnnxRunnerChoice.Default, so => {
                int threads = Preferences.Default.HifiUtauIntraOpThreads;
                if (threads > 0) {
                    so.IntraOpNumThreads = threads;
                }
            });
        }

        private static string PickHnsepModel(string hnsepDir) {
            var pt2 = Directory.GetFiles(hnsepDir, "*.pt2.onnx").FirstOrDefault();
            if (pt2 != null) return pt2;
            return Directory.GetFiles(hnsepDir, "*.onnx").FirstOrDefault()
                ?? throw new FileNotFoundException($"hnsep 目录下未找到 ONNX 模型: {hnsepDir}");
        }

        // ── 预热与重载 ──

        /// <summary>预热：阻塞式加载全部模型（启动预热在后台线程调用）。</summary>
        public void PreloadAll() {
            lock (EngineLock) {
                GetEngine();
            }
        }

        /// <summary>重载整个引擎（两款模型），并清空渲染缓存。</summary>
        public void ReloadAll() {
            lock (EngineLock) {
                lock (loadLock) {
                    ResetSplicer();
                    ResetHnsep();
                    engine = null;   // 必须置空，否则 GetEngine 会返回包装着已释放会话的旧引擎
                }
                GetEngine();
                ClearCache();
            }
        }

        /// <summary>仅重载拼接模型，并清空渲染缓存。</summary>
        public void ReloadSplicer() {
            lock (EngineLock) {
                lock (loadLock) {
                    ResetSplicer();
                    engine = null;
                }
                GetEngine();
                ClearCache();
            }
        }

        /// <summary>仅重载分离模型，并清空渲染缓存。</summary>
        public void ReloadHnsep() {
            lock (EngineLock) {
                lock (loadLock) {
                    ResetHnsep();
                    engine = null;
                }
                GetEngine();
                ClearCache();
            }
        }

        private void ResetSplicer() {
            splicer?.Dispose();
            splicer = null;
            SplicerError = null;
        }

        private void ResetHnsep() {
            hnsep?.Dispose();
            hnsep = null;
            HnsepError = null;
        }

        /// <summary>清空三段式渲染缓存（重载后模型内容可能已变化，缓存键不含模型哈希）。</summary>
        private static void ClearCache() {
            try {
                string root = Path.Combine(PathManager.Inst.CachePath, "hifiutau");
                if (Directory.Exists(root)) {
                    Directory.Delete(root, true);
                }
                foreach (var sub in new[] { "hifigan", "hnsep", "final" }) {
                    Directory.CreateDirectory(Path.Combine(root, sub));
                }
                Log.Information("HiFiUTAU：渲染缓存已清空");
            } catch (Exception e) {
                Log.Error(e, "HiFiUTAU：清空渲染缓存失败");
            }
        }

        /// <summary>vocoder.yaml 子集，避免 HiFi 依赖 DiffSinger 类型。</summary>
        class FusedVocoderYaml {
            public string Model { get; set; } = "model.onnx";
            public int SampleRate { get; set; } = 44100;
            public int HopSize { get; set; } = 512;
            public int NumMelBins { get; set; } = 128;
        }
    }
}
