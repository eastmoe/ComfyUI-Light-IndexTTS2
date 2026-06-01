import platform

from setuptools import find_namespace_packages, setup


setup(
    name="indextts",
    version="0.1.1",
    author="Index SpeechTeam",
    author_email="xuanwu@bilibili.com",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    description="IndexTTS2 core CLI inference package",
    url="https://github.com/index-tts/index-tts",
    packages=find_namespace_packages(include=["indextts", "indextts.*"]),
    include_package_data=True,
    install_requires=[
        "torch>=2.1.2,<2.12",
        "torchaudio>=2.1.2,<2.12",
        "transformers>=4.52.1,<6",
        "tokenizers>=0.21,<1",
        "huggingface_hub",
        "safetensors==0.5.2",
        "modelscope",
        "omegaconf",
        "sentencepiece",
        "librosa>=0.10.2.post1,<0.12",
        "numpy>=1.23,<3",
        "scipy",
        "einops==0.8.1",
        "tqdm",
        "PyYAML",
        "wetext" if platform.system() == "Darwin" else "WeTextProcessing",
    ],
    entry_points={
        "console_scripts": [
            "indextts = indextts.cli:main",
        ]
    },
    license="Apache-2.0",
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
