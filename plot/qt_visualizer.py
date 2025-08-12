import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg


class EpochViewer(QMainWindow):
    """
    A GUI window for visualizing two NumPy array signals epoch by epoch on a single chart (PyQt5 version).
    Supports fixed X-axis range and configurable Y-axis range.

    Parameters:
        signal_data1 (np.ndarray): First signal data.
        signal_data2 (np.ndarray): Second signal data.
        y_range (tuple, optional): A tuple containing (min, max) to fix the Y-axis range.
                                   If None, the Y-axis will auto-scale. Defaults to None.
        parent (QWidget, optional): Parent widget.
    """

    def __init__(self, signal_data1, signal_data2, y_range=None, parent=None):
        super().__init__(parent)

        pg.setConfigOption('antialias', True)

        if signal_data1.shape != signal_data2.shape:
            raise ValueError("The shapes of the two signal arrays must be the same")

        self.data1 = signal_data1
        self.data2 = signal_data2
        self.num_epochs, self.epoch_samples = self.data1.shape
        self.current_epoch_index = 0
        self.y_range = y_range  # Save Y-axis range configuration

        self.setWindowTitle("Epoch Signal Synchronization Visualization Tool")
        self.setGeometry(100, 100, 1000, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self._create_plot(main_layout)
        self._create_controls(main_layout)
        self.update_plot()

    def _create_plot(self, layout):
        """Create a pyqtgraph plot widget and configure axis ranges"""
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.plot_widget = pg.PlotWidget(title="Signal 1 (Blue) vs Signal 2 (Red)")
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Sample Points')
        self.plot_widget.addLegend()

        # Configure axis ranges and mouse behavior
        plot_item = self.plot_widget.getPlotItem()

        # 1. Lock X-axis range from 0 to number of sample points
        plot_item.setXRange(0, self.epoch_samples, padding=0.01)

        # 2. Set Y-axis based on passed parameters
        enable_mouse_y = True
        if self.y_range and isinstance(self.y_range, (list, tuple)) and len(self.y_range) == 2:
            plot_item.setYRange(self.y_range[0], self.y_range[1], padding=0)
            enable_mouse_y = False  # If Y-axis range is fixed, disable its mouse interaction
            self.setWindowTitle(f"{self.windowTitle()} (Y-axis Fixed)")

        # 3. Disable mouse interaction for X-axis, and decide for Y-axis based on conditions
        plot_item.setMouseEnabled(x=False, y=enable_mouse_y)

        layout.addWidget(self.plot_widget)

    def _create_controls(self, layout):
        """Create control buttons and input fields"""
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 10, 0, 10)

        self.prev_button = QPushButton("<< Previous")
        self.prev_button.clicked.connect(self.show_prev_epoch)
        self.next_button = QPushButton("Next >>")
        self.next_button.clicked.connect(self.show_next_epoch)
        self.epoch_label = QLabel()
        self.epoch_label.setAlignment(Qt.AlignCenter)
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("Enter Epoch Number")
        self.jump_input.setFixedWidth(100)
        self.jump_input.returnPressed.connect(self.jump_to_epoch)
        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self.jump_to_epoch)

        controls_layout.addStretch()
        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.epoch_label)
        controls_layout.addWidget(self.next_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.jump_input)
        controls_layout.addWidget(self.jump_button)
        controls_layout.addStretch()

        layout.addWidget(controls_widget)

    def update_plot(self):
        """Update the chart based on the current epoch index"""
        label_text = f"Epoch: {self.current_epoch_index + 1} / {self.num_epochs}"
        self.epoch_label.setText(label_text)
        self.plot_widget.clear()

        pen1 = pg.mkPen(color=(0, 0, 0), width=2)
        self.plot_widget.plot(self.data1[self.current_epoch_index], pen=pen1, name="EEG")
        pen2 = pg.mkPen(color=(200, 0, 0), width=2)
        self.plot_widget.plot(self.data2[self.current_epoch_index], pen=pen2, name="Predict")

        self.prev_button.setEnabled(self.current_epoch_index > 0)
        self.next_button.setEnabled(self.current_epoch_index < self.num_epochs - 1)

    def show_prev_epoch(self):
        if self.current_epoch_index > 0:
            self.current_epoch_index -= 1
            self.update_plot()

    def show_next_epoch(self):
        if self.current_epoch_index < self.num_epochs - 1:
            self.current_epoch_index += 1
            self.update_plot()

    def jump_to_epoch(self):
        try:
            target_epoch = int(self.jump_input.text())
            target_index = target_epoch - 1
            if 0 <= target_index < self.num_epochs:
                self.current_epoch_index = target_index
                self.update_plot()
                self.jump_input.clear()
            else:
                print(f"Error: Please enter an Epoch number between 1 and {self.num_epochs}")
                self.jump_input.selectAll()
        except ValueError:
            print("Error: Invalid input, please enter a number")
            self.jump_input.selectAll()
