#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# CUDA diagnostic and reset utility

import os
import sys

print("="*60)
print("CUDA Diagnostic Tool")
print("="*60)

# Set environment variables for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("\n1. Checking PyTorch installation...")
try:
    import torch
    print(f"   ✓ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"   ✗ PyTorch not found: {e}")
    sys.exit(1)

print("\n2. Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
print(f"   CUDA available: {cuda_available}")

if cuda_available:
    print("\n3. Attempting CUDA initialization...")
    try:
        torch.cuda.init()
        print("   ✓ CUDA initialized successfully")

        print("\n4. Querying CUDA devices...")
        device_count = torch.cuda.device_count()
        print(f"   ✓ Found {device_count} CUDA device(s)")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   Device {i}: {props.name}")
            print(f"   - Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"   - Compute capability: {props.major}.{props.minor}")

        print("\n5. Testing CUDA operations...")
        try:
            # Try to allocate a small tensor
            test_tensor = torch.zeros(10, device='cuda')
            print("   ✓ Successfully allocated test tensor on GPU")

            # Clear cache
            torch.cuda.empty_cache()
            print("   ✓ Cleared CUDA cache")

            # Check memory
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"   - Allocated memory: {allocated:.2f} MB")
            print(f"   - Reserved memory: {reserved:.2f} MB")

        except RuntimeError as e:
            print(f"   ✗ CUDA operation failed: {e}")
            print("\n   Attempting to reset CUDA...")
            try:
                torch.cuda.empty_cache()
                print("   ✓ Cache cleared")
            except Exception as reset_error:
                print(f"   ✗ Reset failed: {reset_error}")

    except RuntimeError as e:
        print(f"   ✗ CUDA initialization failed: {e}")
        print("\n   This might be due to:")
        print("   - Another process using the GPU")
        print("   - Corrupted CUDA state from a previous crash")
        print("   - Driver issues")
        print("\n   Try:")
        print("   - nvidia-smi  # Check GPU status")
        print("   - sudo nvidia-smi --gpu-reset  # Reset GPU (requires root)")
        print("   - Restart your terminal/IDE")

else:
    print("\n   CUDA not available. Possible reasons:")
    print("   - No NVIDIA GPU installed")
    print("   - CUDA toolkit not installed")
    print("   - PyTorch CPU-only version installed")
    print("\n   To check:")
    print("   - nvidia-smi  # Check if GPU is detected")
    print("   - python -c 'import torch; print(torch.version.cuda)'  # Check CUDA version")

print("\n6. Testing caset import...")
try:
    from caset import Spacetime
    print("   ✓ caset imported successfully")
except ImportError as e:
    print(f"   ✗ caset import failed: {e}")
    print("   This is expected if caset hasn't been built/installed yet")

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60)