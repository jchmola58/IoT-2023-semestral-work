"""iot_detector: BENIGN vs MALICIOUS IoT traffic records (CICIoT2023)."""
from .detector import IoTDetector
from .features import FEATURES, TARGET_LIKE_COLUMNS
from .validation import validate_input

__version__ = "0.1.0"
__all__ = ["IoTDetector", "FEATURES", "TARGET_LIKE_COLUMNS", "validate_input", "__version__"]
