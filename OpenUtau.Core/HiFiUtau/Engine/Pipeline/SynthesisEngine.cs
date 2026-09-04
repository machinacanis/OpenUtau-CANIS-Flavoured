using System;
using System.Collections.Generic;
using System.IO;
using NAudio.Wave;
using OpenUtau.Core.HiFiUtau.Engine.Dsp;
using Newtonsoft.Json.Linq;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// synthesis_pipeline/engine.py 的移植：三段式合成编排。
    /// SynthesizeMel  ≙ POST /syn_mel
    /// SynthesizeHnsep ≙ POST /syn_hnsep
    /// SynthesizePost ≙ POST /syn_post
    /// </summary>
    public sealed class SynthesisEngine {
        private readonly HiddenSplicer splicer;
        private readonly Hnsep? hnsep;
        private readonly MelExtractor melExc;

        public SynthesisEngine(HiddenSplicer splicer, Hnsep? hnsep, MelExtractor melExc) {
            this.splicer = splicer;
            this.hnsep = hnsep;
            this.melExc = melExc;
        }

        public int FrontPadFrames => splicer.FrontPadFrames;
        public int TailPadFrames => splicer.TailPadFrames;
        public int ModelHop => splicer.ModelHop;
        public int SampleRate => splicer.SampleRate;
        public Hnsep? HnsepModel => hnsep;

        /// <summary>engine.py _build_fragment：按 SPLC 选择管线（仅解析参数/包络/F0，不切音频）。</summary>
        private PhraseData ParseFragment(JObject jsonData) {
            return PhraseData.Parse(jsonData);
        }

        /// <summary>engine.py _build_fragment + frag.cut_audio：mel 阶段专用（后处理不执行 cut_audio）。</summary>
        private (bool useMel, PhraseData phrase) BuildFragment(JObject jsonData) {
            var phrase = PhraseData.Parse(jsonData);
            if (melExc != null) {
                FragmentMel.CutAudio(phrase, melExc);
            }
            return (melExc != null, phrase);
        }

        /// <summary>engine.py _prepare_mel：phtp → genc → VOL → F0 重采样。</summary>
        public (PhraseData phrase, double[] f0, bool isMel) PrepareMel(JObject jsonData) {
            var (useMel, phrase) = BuildFragment(jsonData);
            if (useMel) {
                FragmentMel.AdjustVolumeByPhtp(phrase);
                FragmentMel.ApplyDynamicGenToMels(phrase);
            } else {
                Fragment.AdjustVolumeByPhtp(phrase, Fragment.MsPerFrame);
                Fragment.ApplyDynamicGenToMels(phrase);
            }

            // VOL：mel += log(gain)
            foreach (var info in phrase.Phones) {
                double vol = info.Flag("vol", 100);
                double gain = vol / 100.0;
                if (Math.Abs(gain - 1.0) > 1e-6 && info.Mel != null && info.MelFrames > 0) {
                    double logGain = Math.Log(gain);
                    for (int i = 0; i < info.Mel.Length; i++) info.Mel[i] += logGain;
                }
            }

            double[] f0 = phrase.GetParam("pit", "pitd");
            f0 = Interpolation.ResampleArray(f0, phrase.HopSize, 512);
            return (phrase, f0, useMel);
        }

        /// <summary>engine.py _synthesize_hifigan。</summary>
        private float[] SynthesizeHifigan(PhraseData phrase, double[] f0, bool isMelPipeline) {
            _ = isMelPipeline;
            return splicer.SpliceAndSynthesizeMel(phrase.Phones, f0);
        }

        /// <summary>POST /syn_mel 等价：mel 拼接 + genc + HiFi-GAN → PCM16 WAV 字节。</summary>
        public byte[] SynthesizeMel(JObject jsonData) {
            var (phrase, f0, isMel) = PrepareMel(jsonData);
            var wav = SynthesizeHifigan(phrase, f0, isMel);
            return WavBytes(wav);
        }

        /// <summary>POST /syn_hnsep 等价：谐波/噪声分离 → (harmonic, noise) PCM16 WAV 字节。</summary>
        public (byte[] harmonic, byte[] noise) SynthesizeHnsep(byte[] wavBytes) {
            if (hnsep == null) {
                throw new InvalidOperationException(
                    "当前合成使用了依赖 HN-SEP 的参数，但 HN-SEP 模型未加载。");
            }
            var wav = WavToSamples(wavBytes);
            var (harmonic, noise) = hnsep.Separate(wav);
            return (WavBytes(ToFloat(harmonic)), WavBytes(ToFloat(noise)));
        }

        /// <summary>POST /syn_post 等价：参数应用 → 最终 PCM16 WAV 字节。</summary>
        public byte[] SynthesizePost(JObject jsonData, byte[]? wavBytes,
                                     byte[]? harmonicBytes, byte[]? noiseBytes) {
            // engine.py synthesize_post：仅解析参数/包络/F0，不执行 cut_audio
            var phrase = ParseFragment(jsonData);
            float[]? wav = null;
            float[]? harmonic = null;
            float[]? noise = null;
            if (harmonicBytes != null && noiseBytes != null) {
                harmonic = WavToSamples(harmonicBytes);
                noise = WavToSamples(noiseBytes);
            } else if (wavBytes != null) {
                wav = WavToSamples(wavBytes);
            } else {
                throw new ArgumentException("SynthesizePost 必须提供 wavBytes 或 (harmonicBytes, noiseBytes)");
            }
            return PostProcess(wav, harmonic, noise, phrase);
        }

        /// <summary>soundfile 'PCM_16' 编码语义（libsndfile: lrint(v*32768) 后截断饱和）。</summary>
        public static byte[] WavBytes(float[] samples) {
            var wav = new MemoryStream();
            using (var w = new NAudio.Wave.WaveFileWriter(new NAudio.Utils.IgnoreDisposeStream(wav),
                       new NAudio.Wave.WaveFormat(44100, 16, 1))) {
                foreach (var s in samples) {
                    double v = Math.Round((double)s * 32768.0, MidpointRounding.ToEven);
                    short pcm = (short)Math.Clamp(v, short.MinValue, short.MaxValue);
                    var bytes = BitConverter.GetBytes(pcm);
                    w.Write(bytes, 0, 2);
                }
            }
            return wav.ToArray();
        }

        internal static float[] ToFloat(double[] x) {
            var f = new float[x.Length];
            for (int i = 0; i < x.Length; i++) f[i] = (float)x[i];
            return f;
        }

        /// <summary>soundfile.read(dtype='float32') 语义。</summary>
        public static float[] WavToSamples(byte[] wavBytes) {
            using var ms = new MemoryStream(wavBytes);
            using var reader = new NAudio.Wave.WaveFileReader(ms);
            var provider = reader.ToSampleProvider().ToMono(1, 0);
            var list = new List<float>(wavBytes.Length / 2);
            var buffer = new float[8192];
            int read;
            while ((read = provider.Read(buffer, 0, buffer.Length)) > 0) {
                for (int i = 0; i < read; i++) list.Add(buffer[i]);
            }
            return list.ToArray();
        }

        /// <summary>engine.py _postprocess —— M3 实现。</summary>
        private byte[] PostProcess(float[]? wav, float[]? harmonic, float[]? noise, PhraseData phrase) {
            return PostProcessing.Postprocess(this, wav, harmonic, noise, phrase);
        }
    }
}
