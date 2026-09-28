import sys

from main import main

if __name__ == "__main__":
    sys.argv = [
        "main.py",
        "--rtsp",
        "--camera",
        "A14-419",
        "--show",
    ]

    sys.exit(main())