"""
SmartKV Package Setup

Supports optional GPU installation with CUDA extensions.
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# GPU requirements
gpu_requires = []
if os.path.exists('requirements-gpu.txt'):
    with open('requirements-gpu.txt') as f:
        gpu_requires = f.read().splitlines()

# Check if CUDA is available and build extensions
# Import torch only when needed to avoid build-time dependency issues
ext_modules = []
cmdclass = {}

# Check for CUDA compiler (not runtime) - works during build
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
cuda_available = os.path.exists(cuda_home)

print(f"[setup.py] CUDA_HOME: {cuda_home}")
print(f"[setup.py] CUDA available: {cuda_available}")

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    print(f"[setup.py] PyTorch version: {torch.__version__}")

    # Build CUDA extensions if compiler is available
    if cuda_available:
        print("[setup.py] Building CUDA extensions...")

        # Use conda's GCC if available
        nvcc_args = [
            '-O3',
            '--use_fast_math',
            '--expt-relaxed-constexpr',
            '-std=c++17',
            # Support multiple GPU architectures
            '-gencode=arch=compute_70,code=sm_70',  # V100
            '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
            '-gencode=arch=compute_80,code=sm_80',  # A100
            '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
            '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
        ]

        # Add ccbin if using conda compiler
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_cxx = os.path.join(conda_prefix, 'bin', 'x86_64-conda-linux-gnu-g++')
            if os.path.exists(conda_cxx):
                nvcc_args = ['-ccbin', conda_cxx] + nvcc_args
                print(f"[setup.py] Using conda g++: {conda_cxx}")

        ext_modules = [
            CUDAExtension(
                name='smartkv_cuda',
                sources=[
                    'smartkv/csrc/quantized_attention.cu',
                    'smartkv/csrc/bit_packing.cu',
                    'smartkv/csrc/bindings.cpp',
                ],
                include_dirs=[
                    'smartkv/csrc',
                ],
                extra_compile_args={
                    'cxx': ['-O3', '-std=c++17'],
                    'nvcc': nvcc_args
                }
            )
        ]
        cmdclass = {'build_ext': BuildExtension}
        print(f"[setup.py] CUDA extensions configured: {len(ext_modules)} module(s)")
    else:
        print("[setup.py] Skipping CUDA extensions - CUDA compiler not found")
except ImportError as e:
    # torch not available during build - skip CUDA extensions
    print(f"[setup.py] Skipping CUDA extensions - torch not available: {e}")
    pass

setup(
    name='smartkv',
    version='0.2.0',
    author='Robby Moseley',
    description='SmartKV: Attention-Guided Adaptive Precision KV-Cache Compression',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/SmartKV',
    packages=find_packages(exclude=['tests', 'scripts', 'venv']),
    install_requires=install_requires,
    extras_require={
        'gpu': gpu_requires,
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
        ]
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
