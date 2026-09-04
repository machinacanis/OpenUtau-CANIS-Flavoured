using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace OpenUtau.Core.HiFiUtau.Engine.Pipeline {
    /// <summary>
    /// synthesis_pipeline.utils.read_audio 的移植：读取音频为 float32 单声道，
    /// 必要时重采样到目标采样率。
    ///
    /// 归一化语义与 soundfile.read(dtype='float32') 对齐：PCM16/24/32 整型除以满量程，
    /// 浮点 WAV 原样。NAudio 的 ToSampleProvider 行为与此一致。
    /// 多声道取均值（soundfile x.mean(axis=1) 语义）。
    /// </summary>
    public static class AudioReader {
        // 同一音源文件会被多个音素重复读取，缓存解码结果
        private static readonly ConcurrentDictionary<string, float[]> Cache = new();

        public static void ClearCache() {
            Cache.Clear();
            audioSampleRates.Clear();
        }

        public static float[] Read(string path, int targetSr = 44100) {
            if (!File.Exists(path)) {
                throw new FileNotFoundException($"音频文件不存在: {path}");
            }
            var audio = Cache.GetOrAdd(path, p => Decode(p));
            if (audio.Length == 0) return Array.Empty<float>();
            // 采样率不匹配时重采样（librosa.resample 语义；用 WdlResampler 实现）
            if (targetSr > 0 && audioSampleRates.TryGetValue(path, out int fs) && fs != targetSr) {
                return Resample(audio, fs, targetSr);
            }
            return audio;
        }

        private static readonly ConcurrentDictionary<string, int> audioSampleRates = new();

        private static float[] Decode(string path) {
            using var reader = new WaveFileReader(path);
            audioSampleRates[path] = reader.WaveFormat.SampleRate;
            var provider = reader.ToSampleProvider().ToMono(1, 0);
            var list = new List<float>(4096);
            var buffer = new float[8192];
            int read;
            while ((read = provider.Read(buffer, 0, buffer.Length)) > 0) {
                for (int i = 0; i < read; i++) list.Add(buffer[i]);
            }
            return list.ToArray();
        }

        public static float[] Resample(float[] input, int fromSr, int toSr) {
            if (fromSr == toSr || input.Length == 0) return input;
            // WDL 重采样（NAudio 内置，纯托管），质量优于线性插值
            var src = new FloatArraySampleProvider(input, fromSr);
            var resampler = new WdlResamplingSampleProvider(src, toSr);
            int outLen = (int)Math.Floor(input.Length * (double)toSr / fromSr);
            var outArr = new float[Math.Max(outLen, 1)];
            int total = 0, read;
            var buffer = new float[8192];
            while (total < outArr.Length && (read = resampler.Read(buffer, 0, Math.Min(buffer.Length, outArr.Length - total))) > 0) {
                Array.Copy(buffer, 0, outArr, total, read);
                total += read;
            }
            if (total < outArr.Length) Array.Resize(ref outArr, total);
            return outArr;
        }

        private sealed class FloatArraySampleProvider : ISampleProvider {
            private readonly float[] data;
            private int position;
            public WaveFormat WaveFormat { get; }
            public FloatArraySampleProvider(float[] data, int sr) {
                this.data = data;
                WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sr, 1);
            }
            public int Read(float[] buffer, int offset, int count) {
                int remaining = data.Length - position;
                if (remaining <= 0) {
                    return 0;
                }
                int n = Math.Min(count, remaining);
                Array.Copy(data, position, buffer, offset, n);
                position += n;
                return n;
            }
        }
    }
}
