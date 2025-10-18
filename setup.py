from setuptools import setup, find_packages

setup(
    name="snapflow",
    version="0.1.0",
    description="Real-time Vision Pipeline for Effects and Games",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["opencv-python>=4.8.0", "mediapipe>=0.10.0", "numpy>=1.24.0"],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "snapflow=snapflow.core.main:main",
        ],
    },
)
