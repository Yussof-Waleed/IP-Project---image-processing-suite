"""
Operations panel widget for organizing processing controls.

Groups operations by category with collapsible sections.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QCheckBox,
    QLabel,
    QHBoxLayout,
    QFrame,
)
from PySide6.QtCore import Signal, Qt


class OperationsPanel(QWidget):
    """
    Panel containing all image processing operations organized by category.
    
    Emits signals when operations are triggered.
    """

    # Signals for operations (category, operation_name, params)
    operation_requested = Signal(str, str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Add operation groups
        self._add_format_conversions()
        self._add_transformations()
        self._add_interpolation()
        self._add_operations()
        self._add_histogram()
        self._add_filters()
        self._add_compression()

        self._container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _create_group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """Create a collapsible group box."""
        group = QGroupBox(title)
        group.setCheckable(False)
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        return group, layout

    def _create_button(self, text: str, category: str, operation: str, 
                       params: dict = None) -> QPushButton:
        """Create an operation button with connected signal."""
        btn = QPushButton(text)
        btn.clicked.connect(
            lambda: self.operation_requested.emit(category, operation, params or {})
        )
        return btn

    # ─────────────────────────────────────────────────────────────────
    # Operation Groups
    # ─────────────────────────────────────────────────────────────────

    def _add_format_conversions(self):
        group, layout = self._create_group("📷 Format Conversions")

        layout.addWidget(self._create_button(
            "Convert to Grayscale", "conversions", "grayscale"
        ))

        # Binary threshold with controls
        binary_frame = QFrame()
        binary_layout = QVBoxLayout(binary_frame)
        binary_layout.setContentsMargins(0, 0, 0, 0)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold:"))
        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(0, 255)
        self._threshold_spin.setValue(128)
        self._threshold_spin.setSpecialValueText("Auto")
        self._threshold_spin.setMinimum(0)
        threshold_row.addWidget(self._threshold_spin)
        binary_layout.addLayout(threshold_row)

        self._invert_check = QCheckBox("Invert")
        binary_layout.addWidget(self._invert_check)

        btn_binary = QPushButton("Apply Binary Threshold")
        btn_binary.clicked.connect(self._on_binary_threshold)
        binary_layout.addWidget(btn_binary)

        btn_evaluate = QPushButton("Evaluate Threshold")
        btn_evaluate.clicked.connect(self._on_evaluate_threshold)
        binary_layout.addWidget(btn_evaluate)

        layout.addWidget(binary_frame)
        self._container_layout.addWidget(group)

    def _add_transformations(self):
        group, layout = self._create_group("🔄 Affine Transformations")

        layout.addWidget(self._create_button(
            "Translation...", "transforms", "translation"
        ))
        layout.addWidget(self._create_button(
            "Scaling...", "transforms", "scaling"
        ))
        layout.addWidget(self._create_button(
            "Rotation...", "transforms", "rotation"
        ))
        layout.addWidget(self._create_button(
            "Shear X...", "transforms", "shear_x"
        ))
        layout.addWidget(self._create_button(
            "Shear Y...", "transforms", "shear_y"
        ))

        self._container_layout.addWidget(group)

    def _add_interpolation(self):
        group, layout = self._create_group("📐 Interpolation / Resize")

        layout.addWidget(self._create_button(
            "Nearest Neighbor...", "interpolation", "nearest"
        ))
        layout.addWidget(self._create_button(
            "Bilinear...", "interpolation", "bilinear"
        ))
        layout.addWidget(self._create_button(
            "Bicubic...", "interpolation", "bicubic"
        ))
        layout.addWidget(self._create_button(
            "Compare All Methods...", "interpolation", "compare"
        ))

        self._container_layout.addWidget(group)

    def _add_operations(self):
        group, layout = self._create_group("✂️ Operations")

        layout.addWidget(self._create_button(
            "Crop (Interactive)...", "operations", "crop_interactive"
        ))
        layout.addWidget(self._create_button(
            "Crop (Coordinates)...", "operations", "crop_coords"
        ))

        self._container_layout.addWidget(group)

    def _add_histogram(self):
        group, layout = self._create_group("📊 Histogram")

        layout.addWidget(self._create_button(
            "Show Histogram", "histogram", "show"
        ))
        layout.addWidget(self._create_button(
            "Analyze Contrast", "histogram", "analyze"
        ))
        layout.addWidget(self._create_button(
            "Equalize Histogram", "histogram", "equalize"
        ))

        self._container_layout.addWidget(group)

    def _add_filters(self):
        group, layout = self._create_group("🎨 Spatial Filters")

        # Low-pass filters
        lp_label = QLabel("<b>Low-Pass (Smoothing)</b>")
        layout.addWidget(lp_label)

        layout.addWidget(self._create_button(
            "Gaussian (19×19, σ=3)", "filters", "gaussian"
        ))
        layout.addWidget(self._create_button(
            "Median (7×7)", "filters", "median"
        ))

        # High-pass filters
        hp_label = QLabel("<b>High-Pass (Edge Detection)</b>")
        layout.addWidget(hp_label)

        layout.addWidget(self._create_button(
            "Laplacian", "filters", "laplacian"
        ))
        layout.addWidget(self._create_button(
            "Sobel", "filters", "sobel"
        ))
        layout.addWidget(self._create_button(
            "Gradient Magnitude", "filters", "gradient"
        ))

        self._container_layout.addWidget(group)

    def _add_compression(self):
        group, layout = self._create_group("📦 Compression")

        layout.addWidget(self._create_button(
            "Huffman Coding", "compression", "huffman"
        ))
        layout.addWidget(self._create_button(
            "Run-Length Encoding", "compression", "rle"
        ))
        layout.addWidget(self._create_button(
            "LZW", "compression", "lzw"
        ))
        layout.addWidget(self._create_button(
            "Arithmetic Coding", "compression", "arithmetic"
        ))
        layout.addWidget(self._create_button(
            "Golomb-Rice", "compression", "golomb"
        ))
        layout.addWidget(self._create_button(
            "Symbol-Based", "compression", "symbol"
        ))
        layout.addWidget(self._create_button(
            "Bit-Plane Encoding", "compression", "bitplane"
        ))
        layout.addWidget(self._create_button(
            "DCT (Block Transform)", "compression", "dct"
        ))
        layout.addWidget(self._create_button(
            "Predictive Coding", "compression", "predictive"
        ))
        layout.addWidget(self._create_button(
            "Wavelet Compression", "compression", "wavelet"
        ))
        layout.addWidget(self._create_button(
            "🔬 Compare All...", "compression", "compare_all"
        ))

        self._container_layout.addWidget(group)

    # ─────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────

    def _on_binary_threshold(self):
        threshold = self._threshold_spin.value()
        # Value 0 means "Auto"
        if threshold == 0:
            threshold = None
        params = {
            "threshold": threshold,
            "invert": self._invert_check.isChecked(),
        }
        self.operation_requested.emit("conversions", "binary", params)

    def _on_evaluate_threshold(self):
        threshold = self._threshold_spin.value()
        if threshold == 0:
            threshold = None
        params = {"threshold": threshold}
        self.operation_requested.emit("conversions", "evaluate_threshold", params)
