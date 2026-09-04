using System;
using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Interactivity;

namespace OpenUtau.App.Controls {
    public partial class UnboundedSlider : UserControl {
        public static readonly StyledProperty<double> ValueProperty =
            AvaloniaProperty.Register<UnboundedSlider, double>(
                nameof(Value), defaultBindingMode: BindingMode.TwoWay);
        public static readonly StyledProperty<double> MinimumProperty =
            AvaloniaProperty.Register<UnboundedSlider, double>(nameof(Minimum), 0);
        public static readonly StyledProperty<double> MaximumProperty =
            AvaloniaProperty.Register<UnboundedSlider, double>(nameof(Maximum), 100);
        public static readonly StyledProperty<double> TickFrequencyProperty =
            AvaloniaProperty.Register<UnboundedSlider, double>(nameof(TickFrequency), 1);
        public static readonly StyledProperty<bool> IsSnapToTickEnabledProperty =
            AvaloniaProperty.Register<UnboundedSlider, bool>(nameof(IsSnapToTickEnabled), true);

        public double Value {
            get => GetValue(ValueProperty);
            set => SetValue(ValueProperty, value);
        }
        public double Minimum {
            get => GetValue(MinimumProperty);
            set => SetValue(MinimumProperty, value);
        }
        public double Maximum {
            get => GetValue(MaximumProperty);
            set => SetValue(MaximumProperty, value);
        }
        public double TickFrequency {
            get => GetValue(TickFrequencyProperty);
            set => SetValue(TickFrequencyProperty, value);
        }
        public bool IsSnapToTickEnabled {
            get => GetValue(IsSnapToTickEnabledProperty);
            set => SetValue(IsSnapToTickEnabledProperty, value);
        }

        public Slider InnerSlider => Slider;
        public TextBox InnerEditor => Editor;

        bool updating;

        public UnboundedSlider() {
            InitializeComponent();
            Slider.ValueChanged += OnSliderValueChanged;
            Editor.LostFocus += OnEditorLostFocus;
            Editor.KeyDown += OnEditorKeyDown;
            SyncFromValue();
        }

        protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change) {
            base.OnPropertyChanged(change);
            if (change.Property == ValueProperty ||
                change.Property == MinimumProperty ||
                change.Property == MaximumProperty) {
                SyncFromValue();
            }
        }

        void OnSliderValueChanged(object? sender, RangeBaseValueChangedEventArgs e) {
            if (updating) {
                return;
            }
            Value = e.NewValue;
        }

        void OnEditorLostFocus(object? sender, RoutedEventArgs e) {
            CommitEditor();
        }

        void OnEditorKeyDown(object? sender, KeyEventArgs e) {
            if (e.Key == Key.Enter) {
                CommitEditor();
                e.Handled = true;
            }
        }

        void CommitEditor() {
            if (double.TryParse(Editor.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out double parsed) ||
                double.TryParse(Editor.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out parsed)) {
                Value = parsed;
            }
            SyncFromValue();
        }

        void SyncFromValue() {
            updating = true;
            try {
                double min = Math.Min(Minimum, Maximum);
                double max = Math.Max(Minimum, Maximum);
                double visual = Math.Clamp(Value, min, max);
                if (Slider.Value != visual) {
                    Slider.Value = visual;
                }
                if (!Editor.IsKeyboardFocusWithin) {
                    Editor.Text = Format(Value);
                }
            } finally {
                updating = false;
            }
        }

        static string Format(double value) {
            if (double.IsNaN(value) || double.IsInfinity(value)) {
                return "0";
            }
            if (Math.Abs(value - Math.Round(value)) < 1e-9) {
                return Math.Round(value).ToString("0", CultureInfo.CurrentCulture);
            }
            return value.ToString("0.###", CultureInfo.CurrentCulture);
        }
    }
}
