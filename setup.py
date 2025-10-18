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

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available():
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
                    'nvcc': [
                        '-O3',
                        '--use_fast_math',
                        '--expt-relaxed-constexpr',
                        # Support multiple GPU architectures
                        '-gencode=arch=compute_70,code=sm_70',  # V100
                        '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
                        '-gencode=arch=compute_80,code=sm_80',  # A100
                        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
                        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
                    ]
                }
            )
        ]
        cmdclass = {'build_ext': BuildExtension}
except ImportError:
    # torch not available during build - skip CUDA extensions
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
