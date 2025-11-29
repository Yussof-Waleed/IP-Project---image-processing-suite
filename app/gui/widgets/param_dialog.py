"""
Parameter input dialogs for operations that need user input.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt

from app.core.interfaces import ParamInfo


class ParameterDialog(QDialog):
    """
    Generic dialog for entering operation parameters.
    
    Dynamically creates input widgets based on ParamInfo definitions.
    """

    def __init__(self, title: str, param_info: dict[str, ParamInfo], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._param_info = param_info
        self._widgets: dict[str, any] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        for name, info in self._param_info.items():
            widget = self._create_widget(info)
            self._widgets[name] = widget
            form.addRow(info.label + ":", widget)
            
            if info.tooltip:
                widget.setToolTip(info.tooltip)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create_widget(self, info: ParamInfo):
        """Create appropriate widget based on parameter type."""
        if info.param_type == "int":
            widget = QSpinBox()
            widget.setRange(
                info.min_val if info.min_val is not None else -10000,
                info.max_val if info.max_val is not None else 10000
            )
            widget.setValue(info.default if info.default is not None else 0)
            return widget
            
        elif info.param_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(
                info.min_val if info.min_val is not None else -10000.0,
                info.max_val if info.max_val is not None else 10000.0
            )
            widget.setSingleStep(0.1)
            widget.setDecimals(2)
            widget.setValue(info.default if info.default is not None else 0.0)
            return widget
            
        elif info.param_type == "bool":
            widget = QCheckBox()
            widget.setChecked(info.default if info.default is not None else False)
            return widget
            
        else:
            # Fallback to label
            return QLabel(str(info.default))

    def get_params(self) -> dict:
        """Get parameter values from widgets."""
        params = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
        return params


class TranslationDialog(ParameterDialog):
    """Dialog for translation parameters."""

    def __init__(self, parent=None):
        param_info = {
            "tx": ParamInfo("X Offset (px)", "int", 0, -2000, 2000, tooltip="Horizontal shift"),
            "ty": ParamInfo("Y Offset (px)", "int", 0, -2000, 2000, tooltip="Vertical shift"),
        }
        super().__init__("Translation", param_info, parent)


class ScalingDialog(ParameterDialog):
    """Dialog for scaling parameters."""

    def __init__(self, parent=None):
        param_info = {
            "sx": ParamInfo("Scale X", "float", 1.0, 0.1, 10.0, tooltip="Horizontal scale factor"),
            "sy": ParamInfo("Scale Y", "float", 1.0, 0.1, 10.0, tooltip="Vertical scale factor"),
            "center": ParamInfo("From Center", "bool", True, tooltip="Scale from image center"),
        }
        super().__init__("Scaling", param_info, parent)


class RotationDialog(ParameterDialog):
    """Dialog for rotation parameters."""

    def __init__(self, parent=None):
        param_info = {
            "angle": ParamInfo("Angle (°)", "float", 0.0, -360.0, 360.0, tooltip="Rotation angle"),
            "expand": ParamInfo("Expand Canvas", "bool", True, tooltip="Expand to fit rotated image"),
        }
        super().__init__("Rotation", param_info, parent)


class ShearDialog(ParameterDialog):
    """Dialog for shear parameters."""

    def __init__(self, direction: str = "X", parent=None):
        param_info = {
            "shear": ParamInfo(f"Shear Factor", "float", 0.0, -2.0, 2.0, 
                              tooltip=f"{'Horizontal' if direction == 'X' else 'Vertical'} shear"),
        }
        super().__init__(f"Shear {direction}", param_info, parent)


class ResizeDialog(ParameterDialog):
    """Dialog for resize/interpolation parameters."""

    def __init__(self, method_name: str = "Resize", parent=None):
        param_info = {
            "scale": ParamInfo("Scale Factor", "float", 2.0, 0.1, 8.0, 
                              tooltip="Scale factor for resizing"),
        }
        super().__init__(f"{method_name} Resize", param_info, parent)


class CropDialog(ParameterDialog):
    """Dialog for crop coordinates."""

    def __init__(self, max_width: int = 10000, max_height: int = 10000, parent=None):
        param_info = {
            "x": ParamInfo("X (left)", "int", 0, 0, max_width, tooltip="Left edge"),
            "y": ParamInfo("Y (top)", "int", 0, 0, max_height, tooltip="Top edge"),
            "width": ParamInfo("Width", "int", 100, 1, max_width, tooltip="Crop width"),
            "height": ParamInfo("Height", "int", 100, 1, max_height, tooltip="Crop height"),
        }
        super().__init__("Crop Region", param_info, parent)
