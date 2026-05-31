import os
from dotenv import load_dotenv
from win32api import GetSystemMetrics

# Tắt MKLDNN nếu có xung đột lỗi
os.environ['FLAGS_use_mkldnn'] = '0'
load_dotenv()

APP_ICON_PATH = "assets/app-logo.ico"          # Window icon (.ico)
SPLASH_LOGO_PATH = "assets/splash-art.jpg"       # Splash logo image

SCREEN_WIDTH = GetSystemMetrics(0)
SCREEN_HEIGHT = GetSystemMetrics(1)

APP_VERSION = "1.0.0"
UPDATE_JSON_URL = "https://raw.githubusercontent.com/Yamakaze-chan/mangalens/main/version.json"