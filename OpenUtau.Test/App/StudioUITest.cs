using OpenUtau.App.Studio;
using OpenUtau.Core.Util;
using Xunit;

namespace OpenUtau.App {
    public class StudioUITest {
        [Fact]
        public void Default_IsClassicEditor() {
            bool previous = Preferences.Default.UseStudioUI;
            string previousTheme = Preferences.Default.ThemeName;
            try {
                Preferences.Default.UseStudioUI = false;
                Assert.False(StudioUI.IsEnabled);
                Assert.Equal(["Light", "Dark"], StudioUI.BuiltInThemes);
                Assert.True(StudioUI.IsBuiltIn("Light"));
                Assert.True(StudioUI.IsBuiltIn("Studio"));
                Assert.DoesNotContain("Studio", StudioUI.BuiltInThemes);
            } finally {
                Preferences.Default.UseStudioUI = previous;
                Preferences.Default.ThemeName = previousTheme;
            }
        }

        [Fact]
        public void Enabled_ExposesStudioThemes() {
            bool previous = Preferences.Default.UseStudioUI;
            string previousTheme = Preferences.Default.ThemeName;
            try {
                Preferences.Default.UseStudioUI = true;
                Assert.True(StudioUI.IsEnabled);
                Assert.Contains("Studio", StudioUI.BuiltInThemes);
                Assert.Contains("WarmSage", StudioUI.BuiltInThemes);
                Assert.True(StudioUI.IsStudioTheme("Studio"));
                Assert.True(StudioUI.IsStudioTheme("WarmSage"));
                Assert.False(StudioUI.IsStudioTheme("Dark"));
            } finally {
                Preferences.Default.UseStudioUI = previous;
                Preferences.Default.ThemeName = previousTheme;
            }
        }

        [Fact]
        public void Disable_DropsStudioPaletteBackToDark() {
            bool previous = Preferences.Default.UseStudioUI;
            string previousTheme = Preferences.Default.ThemeName;
            try {
                Preferences.Default.UseStudioUI = false;
                Preferences.Default.ThemeName = StudioUI.StudioTheme;
                Assert.True(StudioUI.EnsureClassicTheme());
                Assert.Equal("Dark", Preferences.Default.ThemeName);
                Assert.False(StudioUI.EnsureClassicTheme());
            } finally {
                Preferences.Default.UseStudioUI = previous;
                Preferences.Default.ThemeName = previousTheme;
            }
        }
    }
}
