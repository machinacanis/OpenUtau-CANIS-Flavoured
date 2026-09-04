using System.Linq;
using OpenUtau.Core.Util;
using ReactiveUI;

namespace OpenUtau.App.Studio {
    /// <summary>
    /// Studio UI is an opt-in editor chrome, appearance, and (later) layout/tools
    /// layer. Classic OpenUtau editing (NotesViewModel, Commands, hit testing)
    /// stays shared. Studio-only behavior belongs in this namespace — do not
    /// scatter <c>if (UseStudioUI)</c> through ViewModels.
    /// </summary>
    public class StudioUIChangedEvent { }

    public static class StudioUI {
        public const string StudioTheme = "Studio";
        public const string WarmSageTheme = "WarmSage";

        public static readonly string[] ClassicThemes = ["Light", "Dark"];
        public static readonly string[] ExtraThemes = [WarmSageTheme, StudioTheme];

        public static bool IsEnabled => Preferences.Default.UseStudioUI;

        public static bool IsStudioTheme(string? themeName) =>
            themeName == StudioTheme || themeName == WarmSageTheme;

        public static string[] BuiltInThemes =>
            IsEnabled ? [..ClassicThemes, ..ExtraThemes] : ClassicThemes;

        public static bool IsBuiltIn(string themeName) =>
            ClassicThemes.Contains(themeName) || ExtraThemes.Contains(themeName);

        /// <summary>
        /// Drop Studio-only palettes when the user turns Studio UI off.
        /// </summary>
        public static bool EnsureClassicTheme() {
            if (IsEnabled || !IsStudioTheme(Preferences.Default.ThemeName)) {
                return false;
            }
            Preferences.Default.ThemeName = "Dark";
            return true;
        }

        public static void NotifyChanged() {
            MessageBus.Current.SendMessage(new StudioUIChangedEvent());
        }
    }
}
