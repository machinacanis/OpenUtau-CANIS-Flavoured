using System.Collections.Generic;
using Avalonia.Media;
using Avalonia.Media.TextFormatting;

namespace OpenUtau.App.Controls {
    static class TextLayoutCache {
        private static readonly Dictionary<(string text, IBrush brush, double fontSize, string family, FontWeight weight, FontStyle style), TextLayout> cache
            = new Dictionary<(string, IBrush, double, string, FontWeight, FontStyle), TextLayout>();

        public static void Clear() {
            cache.Clear();
        }

        public static TextLayout Get(string text, IBrush brush, double fontSize, bool bold = false) {
            var typeface = new Typeface(
                FontFamily.Default,
                FontStyle.Normal,
                bold ? FontWeight.Bold : FontWeight.Normal);
            return Get(text, brush, fontSize, typeface);
        }

        public static TextLayout Get(string text, IBrush brush, double fontSize, Typeface typeface) {
            string family = typeface.FontFamily.ToString() ?? string.Empty;
            var key = (text, brush, fontSize, family, typeface.Weight, typeface.Style);
            if (!cache.TryGetValue(key, out var textLayout)) {
                textLayout = new TextLayout(
                    text,
                    typeface,
                    fontSize,
                    brush,
                    TextAlignment.Left,
                    TextWrapping.NoWrap);
                cache.Add(key, textLayout);
            }
            return textLayout;
        }
    }
}
