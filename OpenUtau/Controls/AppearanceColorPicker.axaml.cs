using Avalonia;
using Avalonia.Controls;

namespace OpenUtau.App.Controls {
    public partial class AppearanceColorPicker : UserControl {
        public static readonly StyledProperty<int> ModeProperty =
            AvaloniaProperty.Register<AppearanceColorPicker, int>(
                nameof(Mode), defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);
        public static readonly StyledProperty<string> HexProperty =
            AvaloniaProperty.Register<AppearanceColorPicker, string>(
                nameof(Hex), "#FFFFFFFF", defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);
        public static readonly StyledProperty<bool> InvertProperty =
            AvaloniaProperty.Register<AppearanceColorPicker, bool>(
                nameof(Invert), defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);
        public static readonly DirectProperty<AppearanceColorPicker, bool> RgbVisibleProperty =
            AvaloniaProperty.RegisterDirect<AppearanceColorPicker, bool>(nameof(RgbVisible), o => o.RgbVisible);

        public int Mode {
            get => GetValue(ModeProperty);
            set => SetValue(ModeProperty, value);
        }
        public string Hex {
            get => GetValue(HexProperty);
            set => SetValue(HexProperty, value);
        }
        public bool Invert {
            get => GetValue(InvertProperty);
            set => SetValue(InvertProperty, value);
        }
        public bool RgbVisible {
            get => rgbVisible;
            private set => SetAndRaise(RgbVisibleProperty, ref rgbVisible, value);
        }

        private bool rgbVisible;

        public AppearanceColorPicker() {
            InitializeComponent();
            RgbVisible = Mode == 5;
        }

        protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change) {
            base.OnPropertyChanged(change);
            if (change.Property == ModeProperty) {
                RgbVisible = Mode == 5;
            }
        }
    }
}
