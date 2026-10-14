# ===============================================
# MODULE: UTILITY FUNCTIONS
# Fungsi: Helper functions untuk display, FPS calculation, dll
# ===============================================

import cv2
import numpy as np
from typing import Tuple
import config


class FPSCounter:
    """Class untuk menghitung dan tracking Frame Per Second."""
    
    def __init__(self, window_size: int = 30):
        """
        Inisialisasi FPS Counter.
        
        Args:
            window_size: Jumlah frame untuk averaging FPS
        """
        self.frame_times = []
        self.window_size = window_size
        self.prev_time = None
    
    
    def update(self, current_time: float = None) -> float:
        """
        Update FPS counter dengan frame terbaru.
        
        Args:
            current_time: Current timestamp (default: gunakan cv2.getTickCount())
            
        Returns:
            Current FPS
        """
        
        if current_time is None:
            current_time = cv2.getTickCount()
        
        if self.prev_time is not None:
            # Hitung time elapsed dalam milliseconds
            time_elapsed = (current_time - self.prev_time) / cv2.getTickFrequency() * 1000
            self.frame_times.append(time_elapsed)
            
            # Keep only last window_size measurements
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.prev_time = current_time
        
        return self.get_fps()
    
    
    def get_fps(self) -> float:
        """Dapatkan current FPS average."""
        
        if len(self.frame_times) == 0:
            return 0.0
        
        # FPS = 1000 / average_time_per_frame
        avg_time = np.mean(self.frame_times)
        if avg_time > 0:
            return 1000.0 / avg_time
        else:
            return 0.0


class DisplayHelper:
    """Helper class untuk menampilkan informasi di frame."""
    
    @staticmethod
    def put_text(frame: np.ndarray, 
                 text: str, 
                 position: Tuple[int, int],
                 color: Tuple[int, int, int] = config.COLOR_WHITE,
                 font_scale: float = config.FONT_SCALE,
                 thickness: int = config.FONT_THICKNESS,
                 bg_color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Put text pada frame dengan optional background.
        
        Args:
            frame: Input frame
            text: Text yang ingin ditampilkan
            position: Posisi (x, y)
            color: Warna text (BGR)
            font_scale: Ukuran font
            thickness: Ketebalan text
            bg_color: Warna background (optional)
            
        Returns:
            Frame dengan text
        """
        
        frame_copy = frame.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Jika ada background color, gambar rectangle background
        if bg_color is not None:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            x, y = position
            
            # Draw background rectangle
            cv2.rectangle(frame_copy, 
                         (x - 5, y - text_size[1] - 5),
                         (x + text_size[0] + 5, y + 5),
                         bg_color, -1)
        
        # Put text
        cv2.putText(frame_copy, text, position, font, font_scale, color, thickness)
        
        return frame_copy
    
    
    @staticmethod
    def draw_status_panel(frame: np.ndarray,
                         status_dict: dict,
                         position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw panel berisi informasi status di frame.
        
        Args:
            frame: Input frame
            status_dict: Dictionary berisi status items
                        Format: {'Label': 'Value', ...}
            position: Posisi starting panel (x, y)
            
        Returns:
            Frame dengan panel
        """
        
        frame_copy = frame.copy()
        x, y = position
        line_height = 30
        
        for label, value in status_dict.items():
            text = f"{label}: {value}"
            frame_copy = DisplayHelper.put_text(
                frame_copy, text, (x, y),
                color=config.COLOR_WHITE,
                font_scale=config.FONT_SCALE,
                thickness=config.FONT_THICKNESS
            )
            y += line_height
        
        return frame_copy
    
    
    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float,
                position: Tuple[int, int] = None) -> np.ndarray:
        """
        Gambar FPS counter pada frame.
        
        Args:
            frame: Input frame
            fps: Current FPS value
            position: Posisi FPS text (default: top-right)
            
        Returns:
            Frame dengan FPS
        """
        
        if position is None:
            # Default: top-right corner
            h, w = frame.shape[:2]
            position = (w - 150, 30)
        
        text = f"FPS: {fps:.2f}"
        
        return DisplayHelper.put_text(
            frame, text, position,
            color=config.COLOR_GREEN,
            bg_color=config.COLOR_BLACK,
            font_scale=0.6,
            thickness=2
        )
    
    
    @staticmethod
    def draw_detection_box(frame: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          label: str = "",
                          color: Tuple[int, int, int] = config.COLOR_GREEN,
                          thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box dengan label pada frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, width, height)
            label: Label text
            color: Warna box
            thickness: Ketebalan garis
            
        Returns:
            Frame dengan box
        """
        
        frame_copy = frame.copy()
        x, y, w, h = bbox
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label jika ada
        if label:
            frame_copy = DisplayHelper.put_text(
                frame_copy, label, (x, y - 10),
                color=color,
                font_scale=0.6,
                thickness=1
            )
        
        return frame_copy


def normalize_face_image(face_image: np.ndarray, 
                        target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Normalize face image untuk preprocessing (opsional untuk future models).
    
    Args:
        face_image: Input face image
        target_size: Target size untuk resize
        
    Returns:
        Normalized face image
    """
    
    # Resize ke target size
    resized = cv2.resize(face_image, target_size)
    
    # Normalize pixel values ke range [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    return normalized


def log_message(message: str, level: str = "INFO"):
    """
    Print log message dengan level indication.
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
    """
    
    level_colors = {
        "INFO": "\033[94m",      # Blue
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "DEBUG": "\033[92m",     # Green
    }
    
    color = level_colors.get(level, "\033[0m")
    reset = "\033[0m"
    
    print(f"{color}[{level}]{reset} {message}")
