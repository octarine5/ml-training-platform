"""Application entry point for the ML Training Platform."""

from ml_training import __version__


def get_app_info() -> dict:
    """Return application metadata."""
    return {
        "name": "ml-training-platform",
        "version": __version__,
        "description": "ML Training Platform with pipeline parallelism, "
                       "feature engineering, and evaluation systems",
    }


def main() -> None:
    """Run the ML Training Platform."""
    info = get_app_info()
    print(f"{info['name']} v{info['version']}")
    print(info["description"])


if __name__ == "__main__":
    main()
