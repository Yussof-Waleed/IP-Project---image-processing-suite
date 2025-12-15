"""
Interactive crop selection widget.

Allows users to draw a rectangle on an image to select a crop region.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QSizePolicy
)
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor

import numpy as np


class CropSelectionLabel(QLabel):
    """Label that allows drawing a selection rectangle."""
    
    selection_changed = Signal(QRect)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._selection_start = QPoint()
        self._selection_end = QPoint()
        self._is_selecting = False
        self._selection_rect = QRect()
        self._original_pixmap = None
        self._scale_factor = 1.0
        
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def set_image(self, pixmap: QPixmap, scale_factor: float = 1.0):
        """Set the image to display."""
        self._original_pixmap = pixmap
        self._scale_factor = scale_factor
        self._selection_rect = QRect()
        self._update_display()
    
    def _update_display(self):
        """Update the display with current selection overlay."""
        if self._original_pixmap is None:
            return
        
        # Create a copy to draw on
        display = self._original_pixmap.copy()
        
        if not self._selection_rect.isNull():
            painter = QPainter(display)
            
            # Semi-transparent overlay outside selection
            overlay = QColor(0, 0, 0, 120)
            painter.fillRect(display.rect(), overlay)
            
            # Clear the selection area (show original)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self._selection_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            # Draw from original in selection area
            painter.drawPixmap(self._selection_rect, self._original_pixmap, self._selection_rect)
            
            # Draw selection border
            pen = QPen(QColor(0, 120, 215), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._selection_rect)
            
            # Draw corner handles
            handle_size = 8
            painter.setBrush(QColor(0, 120, 215))
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            for corner in self._get_corners():
                painter.drawRect(
                    corner.x() - handle_size // 2,
                    corner.y() - handle_size // 2,
                    handle_size, handle_size
                )
            
            # Draw size label
            if self._selection_rect.width() > 50 and self._selection_rect.height() > 20:
                # Calculate actual image dimensions
                actual_w = int(self._selection_rect.width() / self._scale_factor)
                actual_h = int(self._selection_rect.height() / self._scale_factor)
                size_text = f"{actual_w} × {actual_h}"
                
                painter.setPen(Qt.GlobalColor.white)
                painter.drawText(
                    self._selection_rect.adjusted(5, 5, -5, -5),
                    Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                    size_text
                )
            
            painter.end()
        
        self.setPixmap(display)
    
    def _get_corners(self) -> list:
        """Get corner points of selection rectangle."""
        r = self._selection_rect
        return [
            r.topLeft(), r.topRight(),
            r.bottomLeft(), r.bottomRight()
        ]
    
    def mousePressEvent(self, event):
        """Start selection on mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_selecting = True
            self._selection_start = event.pos()
            self._selection_end = event.pos()
            self._selection_rect = QRect()
            self._update_display()
    
    def mouseMoveEvent(self, event):
        """Update selection during drag."""
        if self._is_selecting:
            self._selection_end = event.pos()
            
            # Clamp to image bounds
            if self._original_pixmap:
                self._selection_end.setX(
                    max(0, min(self._selection_end.x(), self._original_pixmap.width()))
                )
                self._selection_end.setY(
                    max(0, min(self._selection_end.y(), self._original_pixmap.height()))
                )
            
            self._selection_rect = QRect(
                self._selection_start, self._selection_end
            ).normalized()
            
            self._update_display()
    
    def mouseReleaseEvent(self, event):
        """Finish selection on mouse release."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_selecting:
            self._is_selecting = False
            if self._selection_rect.width() > 5 and self._selection_rect.height() > 5:
                self.selection_changed.emit(self._selection_rect)
    
    def get_selection(self) -> tuple:
        """Get the selection rectangle in original image coordinates.
        
        Returns:
            Tuple of (x, y, width, height) in original image pixels, or None if no selection.
        """
        if self._selection_rect.isNull():
            return None
        
        x = int(self._selection_rect.x() / self._scale_factor)
        y = int(self._selection_rect.y() / self._scale_factor)
        w = int(self._selection_rect.width() / self._scale_factor)
        h = int(self._selection_rect.height() / self._scale_factor)
        
        return (x, y, w, h)
    
    def clear_selection(self):
        """Clear the current selection."""
        self._selection_rect = QRect()
        self._update_display()


class InteractiveCropDialog(QDialog):
    """Dialog for interactive crop selection."""
    
    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self._image = image
        self._selection = None
        
        self.setWindowTitle("Interactive Crop")
        self.setMinimumSize(600, 500)
        self.resize(900, 700)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Click and drag to select the crop region. "
            "The selected area will be highlighted."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(instructions)
        
        # Scroll area with crop label
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._crop_label = CropSelectionLabel()
        self._crop_label.selection_changed.connect(self._on_selection_changed)
        scroll.setWidget(self._crop_label)
        
        layout.addWidget(scroll, stretch=1)
        
        # Selection info
        self._info_label = QLabel("No selection")
        self._info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self._info_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self._btn_clear = QPushButton("Clear Selection")
        self._btn_clear.clicked.connect(self._on_clear)
        btn_layout.addWidget(self._btn_clear)
        
        btn_layout.addStretch()
        
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_cancel)
        
        self._btn_crop = QPushButton("Crop")
        self._btn_crop.setEnabled(False)
        self._btn_crop.clicked.connect(self.accept)
        self._btn_crop.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self._btn_crop)
        
        layout.addLayout(btn_layout)
        
        # Load the image
        self._load_image()
    
    def _load_image(self):
        """Load the numpy array into the crop label."""
        arr = self._image
        h, w = arr.shape[:2]
        
        # Ensure uint8
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        
        if arr.ndim == 2:
            qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            bytes_per_line = 3 * w
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale if too large
        max_size = 800
        scale_factor = 1.0
        if pixmap.width() > max_size or pixmap.height() > max_size:
            scale_factor = min(max_size / pixmap.width(), max_size / pixmap.height())
            pixmap = pixmap.scaled(
                int(pixmap.width() * scale_factor),
                int(pixmap.height() * scale_factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        
        self._crop_label.set_image(pixmap, scale_factor)
        self._crop_label.setFixedSize(pixmap.size())
    
    def _on_selection_changed(self, rect: QRect):
        """Handle selection change."""
        selection = self._crop_label.get_selection()
        if selection:
            x, y, w, h = selection
            self._info_label.setText(f"Selection: {w} × {h} pixels at ({x}, {y})")
            self._btn_crop.setEnabled(True)
            self._selection = selection
        else:
            self._info_label.setText("No selection")
            self._btn_crop.setEnabled(False)
            self._selection = None
    
    def _on_clear(self):
        """Clear the selection."""
        self._crop_label.clear_selection()
        self._info_label.setText("No selection")
        self._btn_crop.setEnabled(False)
        self._selection = None
    
    def get_selection(self) -> tuple:
        """Get the final selection.
        
        Returns:
            Tuple of (x, y, width, height) or None.
        """
        return self._selection
