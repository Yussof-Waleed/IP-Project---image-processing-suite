"""
Main application window.

Responsibilities:
- Top-level window layout
- Menu bar and toolbar
- Coordination between panels (delegates to child widgets)
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QStatusBar,
    QSplitter,
    QGroupBox,
    QScrollArea,
    QMessageBox,
    QMenuBar,
    QMenu,
    QCheckBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QAction, QKeySequence

import numpy as np
from pathlib import Path

from app.core.io import load_image, save_image, get_image_metadata, ImageMetadata
from app.core.conversions import GrayscaleProcessor, BinaryThresholdProcessor
from app.core.transforms import (
    TranslationProcessor,
    ScalingProcessor,
    RotationProcessor,
    ShearXProcessor,
    ShearYProcessor,
)
from app.core.interpolation import (
    NearestNeighborResizer,
    BilinearResizer,
    BicubicResizer,
    compare_interpolation_methods,
)
from app.core.histogram import (
    HistogramProcessor,
    HistogramEqualizationProcessor,
)
from app.core.filters import (
    GaussianFilter,
    MedianFilter,
    LaplacianFilter,
    SobelFilter,
    GradientFilter,
)
from app.core.compression import (
    HuffmanCompressor,
    GolombRiceCompressor,
    ArithmeticCompressor,
    LZWCompressor,
    RLECompressor,
    SymbolBasedCompressor,
    BitPlaneCompressor,
    DCTCompressor,
    PredictiveCompressor,
    WaveletCompressor,
    compare_compression,
)
from app.gui.widgets.operations_panel import OperationsPanel
from app.gui.widgets.param_dialog import (
    TranslationDialog,
    ScalingDialog,
    RotationDialog,
    ShearDialog,
    ResizeDialog,
    CropDialog,
)
from app.gui.widgets.crop_widget import InteractiveCropDialog
from app.gui.widgets.histogram_widget import HistogramDialog


class MainWindow(QMainWindow):
    """Main application window with image display and controls."""

    def __init__(self) -> None:
        super().__init__()
        self._current_image: np.ndarray | None = None
        self._processed_image: np.ndarray | None = None
        self._original_image: np.ndarray | None = None  # For reset functionality
        self._current_path: Path | None = None
        self._compound_mode: bool = False  # Apply to processed instead of original
        
        # Initialize processors
        self._grayscale_proc = GrayscaleProcessor()
        self._binary_proc = BinaryThresholdProcessor()
        self._translation_proc = TranslationProcessor()
        self._scaling_proc = ScalingProcessor()
        self._rotation_proc = RotationProcessor()
        self._shear_x_proc = ShearXProcessor()
        self._shear_y_proc = ShearYProcessor()
        self._nn_resizer = NearestNeighborResizer()
        self._bilinear_resizer = BilinearResizer()
        self._bicubic_resizer = BicubicResizer()
        self._histogram_proc = HistogramProcessor()
        self._hist_eq_proc = HistogramEqualizationProcessor()
        self._gaussian_filter = GaussianFilter()
        self._median_filter = MedianFilter()
        self._laplacian_filter = LaplacianFilter()
        self._sobel_filter = SobelFilter()
        self._gradient_filter = GradientFilter()
        
        # Compression processors
        self._huffman = HuffmanCompressor()
        self._golomb = GolombRiceCompressor()
        self._arithmetic = ArithmeticCompressor()
        self._lzw = LZWCompressor()
        self._rle = RLECompressor()
        self._symbol_based = SymbolBasedCompressor()
        self._bitplane = BitPlaneCompressor()
        self._dct = DCTCompressor()
        self._predictive = PredictiveCompressor()
        self._wavelet = WaveletCompressor()
        
        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()

    def _setup_menubar(self) -> None:
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_load_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_image)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Image Processing Mini-Suite")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: Controls
        controls_panel = self._create_controls_panel()
        main_layout.addWidget(controls_panel, stretch=0)

        # Right panel: Image display (splitter for original + processed)
        image_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self._original_display = self._create_image_display("Original Image")
        self._processed_display = self._create_image_display("Processed Image")
        
        image_splitter.addWidget(self._original_display)
        image_splitter.addWidget(self._processed_display)
        
        main_layout.addWidget(image_splitter, stretch=1)

    def _create_controls_panel(self) -> QWidget:
        """Create the left-side controls panel."""
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # File operations group
        file_group = QGroupBox("📁 File Operations")
        file_layout = QVBoxLayout(file_group)
        
        self._btn_load = QPushButton("📂 Load Image")
        self._btn_load.clicked.connect(self._on_load_image)
        file_layout.addWidget(self._btn_load)
        
        self._btn_save = QPushButton("💾 Save Result")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save_image)
        file_layout.addWidget(self._btn_save)
        
        # Reset button
        self._btn_reset = QPushButton("🔄 Reset to Original")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._on_reset_to_original)
        file_layout.addWidget(self._btn_reset)
        
        # Compound mode checkbox
        self._chk_compound = QCheckBox("🔗 Compound Modifications")
        self._chk_compound.setToolTip(
            "When enabled, operations apply to the processed image\n"
            "instead of the original, allowing stacked effects."
        )
        self._chk_compound.toggled.connect(self._on_compound_toggled)
        file_layout.addWidget(self._chk_compound)
        
        layout.addWidget(file_group)

        # Metadata display
        self._metadata_group = QGroupBox("ℹ️ Image Info")
        metadata_layout = QVBoxLayout(self._metadata_group)
        self._lbl_metadata = QLabel("No image loaded")
        self._lbl_metadata.setWordWrap(True)
        metadata_layout.addWidget(self._lbl_metadata)
        layout.addWidget(self._metadata_group)

        # Operations panel
        self._operations_panel = OperationsPanel()
        self._operations_panel.operation_requested.connect(self._on_operation_requested)
        layout.addWidget(self._operations_panel, stretch=1)

        return panel

    def _create_image_display(self, title: str) -> QWidget:
        """Create a scrollable image display widget."""
        container = QGroupBox(title)
        layout = QVBoxLayout(container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setText("No image")
        label.setStyleSheet("background-color: #2b2b2b; color: #888;")
        scroll.setWidget(label)

        layout.addWidget(scroll)

        # Store reference to label for updates
        container.image_label = label
        return container

    def _setup_statusbar(self) -> None:
        """Create status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

    # ─────────────────────────────────────────────────────────────────
    # Event Handlers (Commands)
    # ─────────────────────────────────────────────────────────────────

    def _on_load_image(self) -> None:
        """Handle load image button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;All Files (*)"
        )
        if not file_path:
            return
        
        self._load_and_display_image(Path(file_path))

    def _load_and_display_image(self, path: Path) -> None:
        """Load image from path and update displays."""
        try:
            self._current_image = load_image(path)
            self._original_image = self._current_image.copy()  # Save original for reset
            self._current_path = path
            self._processed_image = None
            
            # Update original display
            self._display_array(self._current_image, self._original_display.image_label)
            
            # Clear processed display
            self._processed_display.image_label.clear()
            self._processed_display.image_label.setText("No processing applied")
            
            # Enable/disable buttons
            self._btn_reset.setEnabled(False)  # No processing yet
            
            # Update metadata
            metadata = get_image_metadata(path)
            self._update_metadata_display(metadata)
            
            self._statusbar.showMessage(f"Loaded: {path.name}")
            
        except Exception as e:
            self._statusbar.showMessage(f"Error loading image: {e}")

    def _on_compound_toggled(self, checked: bool) -> None:
        """Handle compound mode checkbox toggle."""
        self._compound_mode = checked
        if checked:
            self._statusbar.showMessage("🔗 Compound mode: Operations will stack on processed image")
        else:
            self._statusbar.showMessage("🔗 Compound mode off: Operations apply to original image")

    def _on_reset_to_original(self) -> None:
        """Reset to the original loaded image."""
        if self._original_image is not None:
            self._current_image = self._original_image.copy()
            self._processed_image = None
            
            # Update displays
            self._display_array(self._current_image, self._original_display.image_label)
            self._processed_display.image_label.clear()
            self._processed_display.image_label.setText("No processing applied")
            
            self._btn_reset.setEnabled(False)
            self._btn_save.setEnabled(False)
            self._statusbar.showMessage("🔄 Reset to original image")

    def _get_source_image(self) -> np.ndarray:
        """Get the image to apply operations to based on compound mode."""
        if self._compound_mode and self._processed_image is not None:
            return self._processed_image
        return self._current_image

    # ─────────────────────────────────────────────────────────────────
    # Queries (read-only helpers)
    # ─────────────────────────────────────────────────────────────────

    def _display_array(self, arr: np.ndarray, label: QLabel) -> None:
        """Convert numpy array to QPixmap and display on label."""
        if arr is None:
            label.clear()
            return

        # Ensure uint8
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        h, w = arr.shape[:2]
        
        if arr.ndim == 2:
            # Grayscale
            qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # Color (RGB)
            bytes_per_line = 3 * w
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        
        # Scale if too large, preserving aspect ratio
        max_size = 800
        if pixmap.width() > max_size or pixmap.height() > max_size:
            pixmap = pixmap.scaled(
                max_size, max_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        
        label.setPixmap(pixmap)

    def _update_metadata_display(self, metadata: ImageMetadata) -> None:
        """Update the metadata label with image info."""
        text = (
            f"<b>Resolution:</b> {metadata.width} × {metadata.height} px<br>"
            f"<b>Channels:</b> {metadata.channels}<br>"
            f"<b>File Size:</b> {metadata.file_size_str}<br>"
            f"<b>Format:</b> {metadata.format}"
        )
        self._lbl_metadata.setText(text)

    # ─────────────────────────────────────────────────────────────────
    # Operation Dispatcher
    # ─────────────────────────────────────────────────────────────────

    def _on_operation_requested(self, category: str, operation: str, params: dict) -> None:
        """Handle operation requests from the operations panel."""
        if self._current_image is None:
            self._statusbar.showMessage("Please load an image first")
            return

        try:
            if category == "conversions":
                self._handle_conversion(operation, params)
            elif category == "transforms":
                self._handle_transform(operation, params)
            elif category == "interpolation":
                self._handle_interpolation(operation, params)
            elif category == "operations":
                self._handle_operations(operation, params)
            elif category == "histogram":
                self._handle_histogram(operation, params)
            elif category == "filters":
                self._handle_filter(operation, params)
            elif category == "compression":
                self._handle_compression(operation, params)
            else:
                self._statusbar.showMessage(f"Unknown category: {category}")
        except Exception as e:
            self._statusbar.showMessage(f"Error: {e}")
            QMessageBox.warning(self, "Processing Error", str(e))

    def _handle_conversion(self, operation: str, params: dict) -> None:
        """Handle format conversion operations."""
        source = self._get_source_image()
        
        if operation == "grayscale":
            result = self._grayscale_proc.process(source)
            self.set_processed_image(result)
            self._statusbar.showMessage("Converted to grayscale")

        elif operation == "binary":
            result = self._binary_proc.process(source, **params)
            self.set_processed_image(result)
            threshold = params.get("threshold", "auto")
            self._statusbar.showMessage(f"Applied binary threshold: {threshold}")

        elif operation == "evaluate_threshold":
            evaluation = self._binary_proc.evaluate_threshold(source)
            
            # Check if image is already binary
            if evaluation.get('is_binary', False):
                QMessageBox.information(self, "Threshold Evaluation",
                    f"<h3>ℹ️ Image is Already Binary</h3>"
                    f"<p>Foreground: {evaluation['foreground_ratio']*100:.0f}% | "
                    f"Background: {evaluation['background_ratio']*100:.0f}%</p>"
                )
                return
            
            # Show evaluation result
            is_optimal = evaluation['is_optimal']
            mean_t = evaluation['mean_threshold']
            otsu_t = evaluation['otsu_threshold']
            quality = evaluation['mean_quality_percent']
            
            color = "#4CAF50" if is_optimal else "#F44336"
            status = "✅ OPTIMAL" if is_optimal else "❌ NOT OPTIMAL"
            
            html = (
                f"<h3 style='color:{color};'>{status}</h3>"
                f"<table cellpadding='4'>"
                f"<tr><td>Mean Threshold:</td><td><b>{mean_t:.0f}</b></td></tr>"
                f"<tr><td>Otsu Threshold:</td><td><b>{otsu_t:.0f}</b></td></tr>"
                f"<tr><td>Quality:</td><td>{quality:.0f}%</td></tr>"
                f"<tr><td>Foreground:</td><td>{evaluation['foreground_ratio']*100:.0f}%</td></tr>"
                f"<tr><td>Background:</td><td>{evaluation['background_ratio']*100:.0f}%</td></tr>"
                f"</table>"
            )
            if not is_optimal:
                html += f"<p><b>💡 Use Otsu ({otsu_t:.0f}) for better results.</b></p>"
            
            QMessageBox.information(self, "Threshold Evaluation", html)

    def _handle_transform(self, operation: str, params: dict) -> None:
        """Handle affine transformation operations."""
        source = self._get_source_image()
        
        if operation == "translation":
            dialog = TranslationDialog(self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._translation_proc.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Translated by ({params['tx']}, {params['ty']})")

        elif operation == "scaling":
            dialog = ScalingDialog(self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._scaling_proc.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Scaled by ({params['sx']:.2f}, {params['sy']:.2f})")

        elif operation == "rotation":
            dialog = RotationDialog(self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._rotation_proc.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Rotated by {params['angle']:.1f}°")

        elif operation == "shear_x":
            dialog = ShearDialog("X", self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._shear_x_proc.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Sheared X by {params['shear']:.2f}")

        elif operation == "shear_y":
            dialog = ShearDialog("Y", self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._shear_y_proc.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Sheared Y by {params['shear']:.2f}")

    def _handle_interpolation(self, operation: str, params: dict) -> None:
        """Handle interpolation/resize operations."""
        source = self._get_source_image()
        
        if operation == "nearest":
            dialog = ResizeDialog("Nearest Neighbor", self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._nn_resizer.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Resized (NN) by {params['scale']:.2f}x")

        elif operation == "bilinear":
            dialog = ResizeDialog("Bilinear", self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._bilinear_resizer.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Resized (Bilinear) by {params['scale']:.2f}x")

        elif operation == "bicubic":
            dialog = ResizeDialog("Bicubic", self)
            if dialog.exec():
                params = dialog.get_params()
                result = self._bicubic_resizer.process(source, **params)
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Resized (Bicubic) by {params['scale']:.2f}x")

        elif operation == "compare":
            dialog = ResizeDialog("Compare Methods", self)
            if dialog.exec():
                params = dialog.get_params()
                scale = params.get("scale", 2.0)
                results = compare_interpolation_methods(source, scale)
                
                # Show bicubic result (highest quality)
                self.set_processed_image(results["Bicubic"])
                self._statusbar.showMessage(f"Compared methods at {scale:.2f}x (showing Bicubic)")
                
                # Show info about all methods
                h, w = source.shape[:2]
                new_size = f"{int(w*scale)}x{int(h*scale)}"
                QMessageBox.information(self, "Interpolation Comparison",
                    f"<b>Resized from {w}x{h} to {new_size}</b><br><br>"
                    f"• <b>Nearest Neighbor:</b> Fast, pixelated<br>"
                    f"• <b>Bilinear:</b> Smooth, some blur<br>"
                    f"• <b>Bicubic:</b> Best quality (shown)<br>")

    def _handle_operations(self, operation: str, params: dict) -> None:
        """Handle general operations (crop, etc.)."""
        source = self._get_source_image()
        
        if operation == "crop_coords":
            h, w = source.shape[:2]
            dialog = CropDialog(w, h, self)
            if dialog.exec():
                params = dialog.get_params()
                x, y = params["x"], params["y"]
                cw, ch = params["width"], params["height"]
                
                # Validate bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                cw = min(cw, w - x)
                ch = min(ch, h - y)
                
                if source.ndim == 3:
                    result = source[y:y+ch, x:x+cw, :].copy()
                else:
                    result = source[y:y+ch, x:x+cw].copy()
                
                self.set_processed_image(result)
                self._statusbar.showMessage(f"Cropped to {cw}x{ch} at ({x}, {y})")

        elif operation == "crop_interactive":
            dialog = InteractiveCropDialog(source, self)
            if dialog.exec():
                selection = dialog.get_selection()
                if selection:
                    x, y, cw, ch = selection
                    if source.ndim == 3:
                        result = source[y:y+ch, x:x+cw, :].copy()
                    else:
                        result = source[y:y+ch, x:x+cw].copy()
                    self.set_processed_image(result)
                    self._statusbar.showMessage(f"Cropped to {cw}×{ch} at ({x}, {y})")
            else:
                self._statusbar.showMessage("Crop cancelled")

    def _handle_histogram(self, operation: str, params: dict) -> None:
        """Handle histogram operations."""
        source = self._get_source_image()
        
        if operation == "show":
            result = self._histogram_proc.compute_histogram(source)
            dialog = HistogramDialog(source, result, self)
            dialog.exec()
            self._statusbar.showMessage("Histogram displayed")

        elif operation == "analyze":
            result = self._histogram_proc.compute_histogram(source)
            status = "⚠️ Low Contrast" if result.is_low_contrast else "✓ Good Contrast"
            info = (
                f"<b>Contrast Analysis</b><br><br>"
                f"<b>Status:</b> {status}<br><br>"
                f"<b>Analysis:</b><br>{result.contrast_analysis}<br><br>"
                f"<b>Statistics:</b><br>"
                f"Mean: {result.mean:.1f}<br>"
                f"Std Dev: {result.std:.1f}<br>"
                f"Dynamic Range: {result.max_val - result.min_val} levels"
            )
            QMessageBox.information(self, "Contrast Analysis", info)
            self._statusbar.showMessage(f"Contrast: {'Low' if result.is_low_contrast else 'Good'}")

        elif operation == "equalize":
            result = self._hist_eq_proc.process(source)
            self.set_processed_image(result)
            self._statusbar.showMessage("Histogram equalization applied")

    def _handle_filter(self, operation: str, params: dict) -> None:
        """Handle spatial filter operations."""
        source = self._get_source_image()
        
        if operation == "gaussian":
            result = self._gaussian_filter.process(source)
            self.set_processed_image(result)
            self._statusbar.showMessage("Applied Gaussian filter (19×19, σ=3)")

        elif operation == "median":
            result = self._median_filter.process(source)
            self.set_processed_image(result)
            self._statusbar.showMessage("Applied Median filter (7×7)")

        elif operation == "laplacian":
            result = self._laplacian_filter.process(source)
            self.set_processed_image(result)
            self._statusbar.showMessage("Applied Laplacian edge detection")

        elif operation == "sobel":
            result = self._sobel_filter.process(source)
            self.set_processed_image(result)
            self._statusbar.showMessage("Applied Sobel edge detection")

        elif operation == "gradient":
            result = self._gradient_filter.process(source)
            self.set_processed_image(result)
            self._statusbar.showMessage("Applied Gradient magnitude")

    def _handle_compression(self, operation: str, params: dict) -> None:
        """Handle compression operations."""
        source = self._get_source_image()
        
        compressors = {
            "huffman": (self._huffman, "Huffman"),
            "golomb": (self._golomb, "Golomb-Rice"),
            "arithmetic": (self._arithmetic, "Arithmetic"),
            "lzw": (self._lzw, "LZW"),
            "rle": (self._rle, "RLE"),
            "symbol": (self._symbol_based, "Symbol-Based"),
            "bitplane": (self._bitplane, "Bit-Plane"),
            "dct": (self._dct, "DCT"),
            "predictive": (self._predictive, "Predictive"),
            "wavelet": (self._wavelet, "Wavelet"),
        }
        
        if operation == "compare_all":
            results = compare_compression(source)
            info_text = "<b>Compression Comparison</b><br><br>"
            info_text += "<table border='1' cellpadding='4'>"
            info_text += "<tr><th>Algorithm</th><th>Original</th><th>Compressed</th><th>Ratio</th></tr>"
            
            for r in results:
                info_text += (
                    f"<tr><td>{r.algorithm}</td>"
                    f"<td>{r.original_size:,} B</td>"
                    f"<td>{r.compressed_size:,} B</td>"
                    f"<td><b>{r.compression_ratio:.2f}×</b></td></tr>"
                )
            info_text += "</table>"
            
            best = results[0] if results else None
            if best:
                info_text += f"<br><br><b>Best:</b> {best.algorithm} with {best.compression_ratio:.2f}× ratio"
            
            QMessageBox.information(self, "Compression Results", info_text)
            self._statusbar.showMessage("Compared all 10 compression algorithms")
            return
        
        if operation in compressors:
            compressor, name = compressors[operation]
            result = compressor.compress(source)
            
            # Generate preview image
            # Lossless compressions: preview is the original (perfect reconstruction)
            # Lossy compressions (DCT, Wavelet): show quality degradation
            if operation in ("dct", "wavelet"):
                # For lossy, show a simulated degraded version
                preview = self._generate_lossy_preview(operation)
            else:
                # Lossless: preview is same as original
                preview = source.copy()
            
            # Show the preview
            self.set_processed_image(preview)
            
            info_text = (
                f"<b>{name} Compression</b><br><br>"
                f"<b>Original Size:</b> {result.original_size:,} bytes<br>"
                f"<b>Compressed Size:</b> {result.compressed_size:,} bytes<br>"
                f"<b>Compression Ratio:</b> {result.compression_ratio:.2f}×<br>"
                f"<b>Space Savings:</b> {(1 - 1/result.compression_ratio)*100:.1f}%<br><br>"
            )
            
            if result.metadata:
                info_text += "<b>Details:</b><br>"
                for key, val in result.metadata.items():
                    info_text += f"• {key}: {val}<br>"
            
            info_text += "<br><i>Preview shown on right (save enabled)</i>"
            
            QMessageBox.information(self, f"{name} Results", info_text)
            self._statusbar.showMessage(f"{name}: {result.compression_ratio:.2f}× ratio - Preview shown")
    
    def _generate_lossy_preview(self, method: str) -> np.ndarray:
        """Generate preview showing quality loss for lossy compression."""
        source = self._get_source_image()
        gray = source if source.ndim == 2 else \
               np.mean(source, axis=2).astype(np.float64)
        h, w = gray.shape
        
        if method == "dct":
            # Simulate JPEG-like quality loss with block artifacts
            result = np.zeros_like(gray)
            quant_factor = 8  # Higher = more loss
            
            # Process 8x8 blocks
            for i in range(0, h - h % 8, 8):
                for j in range(0, w - w % 8, 8):
                    block = gray[i:i+8, j:j+8]
                    # Simulate quantization loss
                    quantized = np.round(block / quant_factor) * quant_factor
                    result[i:i+8, j:j+8] = quantized
            
            # Handle edge pixels
            result[h - h % 8:, :] = gray[h - h % 8:, :]
            result[:, w - w % 8:] = gray[:, w - w % 8:]
            
            return np.clip(result, 0, 255).astype(np.uint8)
        
        elif method == "wavelet":
            # Simulate wavelet thresholding loss (softer blur)
            kernel_size = 3
            result = np.zeros_like(gray)
            
            for i in range(h):
                for j in range(w):
                    # Average with neighbors
                    i_min = max(0, i - 1)
                    i_max = min(h, i + 2)
                    j_min = max(0, j - 1)
                    j_max = min(w, j + 2)
                    result[i, j] = np.mean(gray[i_min:i_max, j_min:j_max])
            
            return np.clip(result, 0, 255).astype(np.uint8)
        
        return source.copy()

    def _on_save_image(self) -> None:
        """Save the processed image to disk."""
        if self._processed_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)"
        )
        if not file_path:
            return

        try:
            save_image(self._processed_image, Path(file_path))
            self._statusbar.showMessage(f"Saved: {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    # ─────────────────────────────────────────────────────────────────
    # Public API for processing (will be extended)
    # ─────────────────────────────────────────────────────────────────

    def get_current_image(self) -> np.ndarray | None:
        """Return the currently loaded image array (query)."""
        return self._current_image

    def set_processed_image(self, image: np.ndarray) -> None:
        """Set and display the processed result (command)."""
        self._processed_image = image
        self._display_array(image, self._processed_display.image_label)
        self._btn_save.setEnabled(True)
        self._btn_reset.setEnabled(True)
