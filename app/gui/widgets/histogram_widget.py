"""
Histogram visualization widget.

Displays histogram as a graphical chart with statistics overlay.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QSizePolicy, QGroupBox, QTabWidget
)
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QPainterPath

import numpy as np

from app.core.interfaces import HistogramResult


class HistogramCanvas(QWidget):
    """Widget that draws a histogram chart."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._histogram = None
        self._color = QColor(100, 150, 200)
        self._title = "Histogram"
        
        self.setMinimumSize(400, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    def set_histogram(self, histogram: np.ndarray, color: QColor = None, title: str = None):
        """Set the histogram data to display."""
        self._histogram = histogram
        if color:
            self._color = color
        if title:
            self._title = title
        self.update()
    
    def paintEvent(self, event):
        """Draw the histogram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 35))
        
        if self._histogram is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No histogram data")
            return
        
        # Margins
        margin_left = 50
        margin_right = 20
        margin_top = 30
        margin_bottom = 40
        
        chart_width = self.width() - margin_left - margin_right
        chart_height = self.height() - margin_top - margin_bottom
        
        if chart_width <= 0 or chart_height <= 0:
            return
        
        # Draw title
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(
            QRect(margin_left, 5, chart_width, 20),
            Qt.AlignmentFlag.AlignCenter,
            self._title
        )
        
        # Draw axes
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        # X axis
        painter.drawLine(
            margin_left, self.height() - margin_bottom,
            self.width() - margin_right, self.height() - margin_bottom
        )
        # Y axis
        painter.drawLine(
            margin_left, margin_top,
            margin_left, self.height() - margin_bottom
        )
        
        # Normalize histogram
        max_val = self._histogram.max()
        if max_val == 0:
            return
        
        normalized = self._histogram / max_val
        
        # Draw histogram bars
        bar_width = chart_width / 256
        
        # Create gradient fill
        for i in range(256):
            bar_height = int(normalized[i] * chart_height)
            if bar_height < 1:
                continue
            
            x = margin_left + i * bar_width
            y = self.height() - margin_bottom - bar_height
            
            # Color intensity based on bin value
            intensity = i / 255
            color = QColor(
                int(self._color.red() * (0.5 + 0.5 * intensity)),
                int(self._color.green() * (0.5 + 0.5 * intensity)),
                int(self._color.blue() * (0.5 + 0.5 * intensity))
            )
            
            painter.fillRect(
                int(x), int(y),
                max(1, int(bar_width)), bar_height,
                color
            )
        
        # Draw X axis labels
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont("Arial", 8))
        for val in [0, 64, 128, 192, 255]:
            x = margin_left + (val / 255) * chart_width
            painter.drawText(
                int(x - 15), self.height() - margin_bottom + 5,
                30, 20,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                str(val)
            )
        
        # Draw Y axis label
        painter.save()
        painter.translate(15, self.height() // 2)
        painter.rotate(-90)
        painter.drawText(
            -40, 0, 80, 20,
            Qt.AlignmentFlag.AlignCenter,
            "Frequency"
        )
        painter.restore()
        
        # X axis label
        painter.drawText(
            margin_left, self.height() - 15,
            chart_width, 15,
            Qt.AlignmentFlag.AlignCenter,
            "Intensity"
        )


class HistogramDialog(QDialog):
    """Dialog displaying histogram with statistics."""
    
    def __init__(self, image: np.ndarray, result: HistogramResult = None, parent=None):
        super().__init__(parent)
        self._image = image
        self._result = result
        
        self.setWindowTitle("Histogram Analysis")
        self.setMinimumSize(600, 500)
        self.resize(700, 550)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Check if color or grayscale
        is_color = self._image.ndim == 3
        
        if is_color:
            # Tabbed view for color images
            tabs = QTabWidget()
            
            # Combined/Luminance histogram
            lum_canvas = HistogramCanvas()
            gray = np.mean(self._image, axis=2).astype(np.uint8)
            lum_hist = np.bincount(gray.flatten(), minlength=256)
            lum_canvas.set_histogram(lum_hist, QColor(200, 200, 200), "Luminance")
            tabs.addTab(lum_canvas, "Luminance")
            
            # Red channel
            r_canvas = HistogramCanvas()
            r_hist = np.bincount(self._image[:, :, 0].flatten(), minlength=256)
            r_canvas.set_histogram(r_hist, QColor(220, 80, 80), "Red Channel")
            tabs.addTab(r_canvas, "Red")
            
            # Green channel
            g_canvas = HistogramCanvas()
            g_hist = np.bincount(self._image[:, :, 1].flatten(), minlength=256)
            g_canvas.set_histogram(g_hist, QColor(80, 200, 80), "Green Channel")
            tabs.addTab(g_canvas, "Green")
            
            # Blue channel
            b_canvas = HistogramCanvas()
            b_hist = np.bincount(self._image[:, :, 2].flatten(), minlength=256)
            b_canvas.set_histogram(b_hist, QColor(80, 120, 220), "Blue Channel")
            tabs.addTab(b_canvas, "Blue")
            
            layout.addWidget(tabs, stretch=1)
        else:
            # Single histogram for grayscale
            canvas = HistogramCanvas()
            hist = np.bincount(self._image.flatten().astype(np.uint8), minlength=256)
            canvas.set_histogram(hist, QColor(150, 180, 220), "Grayscale Histogram")
            layout.addWidget(canvas, stretch=1)
        
        # Statistics panel
        stats_group = QGroupBox("📊 Statistics")
        stats_layout = QHBoxLayout(stats_group)
        
        if self._result:
            # Left column
            left_stats = QLabel(
                f"<b>Mean:</b> {self._result.mean:.1f}<br>"
                f"<b>Std Dev:</b> {self._result.std:.1f}<br>"
                f"<b>Min:</b> {self._result.min_val}"
            )
            stats_layout.addWidget(left_stats)
            
            # Right column
            right_stats = QLabel(
                f"<b>Max:</b> {self._result.max_val}<br>"
                f"<b>Range:</b> {self._result.max_val - self._result.min_val}<br>"
                f"<b>Contrast:</b> {'⚠️ Low' if self._result.is_low_contrast else '✓ Good'}"
            )
            stats_layout.addWidget(right_stats)
        else:
            # Compute basic stats
            if is_color:
                gray = np.mean(self._image, axis=2)
            else:
                gray = self._image
            
            mean = np.mean(gray)
            std = np.std(gray)
            min_val = int(np.min(gray))
            max_val = int(np.max(gray))
            
            left_stats = QLabel(
                f"<b>Mean:</b> {mean:.1f}<br>"
                f"<b>Std Dev:</b> {std:.1f}<br>"
                f"<b>Min:</b> {min_val}"
            )
            stats_layout.addWidget(left_stats)
            
            right_stats = QLabel(
                f"<b>Max:</b> {max_val}<br>"
                f"<b>Range:</b> {max_val - min_val}<br>"
                f"<b>Pixels:</b> {gray.size:,}"
            )
            stats_layout.addWidget(right_stats)
        
        layout.addWidget(stats_group)
        
        # Analysis text if available
        if self._result and self._result.contrast_analysis:
            analysis_label = QLabel(f"<i>{self._result.contrast_analysis}</i>")
            analysis_label.setWordWrap(True)
            analysis_label.setStyleSheet("padding: 5px; color: #aaa;")
            layout.addWidget(analysis_label)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
