import subprocess
import warnings


def check_ffmpeg():
    """Checks if ffmpeg is available on the system and outputs a warning if not."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn("ffmpeg is not found. Some functionalities might be limited.")
