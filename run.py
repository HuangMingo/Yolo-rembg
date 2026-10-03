from app.config import Settings
from app.ui import launch

APP_SETTINGS = Settings()


if __name__ == "__main__":
    launch(settings=APP_SETTINGS)
