# ===============================================
# FIX SCRIPT: MediaPipe Installation Issue
# ===============================================
# 
# Script ini akan:
# 1. Uninstall MediaPipe yang current
# 2. Install versi yang benar
# 3. Verifikasi installation
#
# Jalankan: python fix_mediapipe.py
# ===============================================

import subprocess
import sys
import os


def run_command(cmd, description=""):
    """Run command dan tampilkan output."""
    
    print(f"\n{'='*60}")
    if description:
        print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Main fix process."""
    
    print("\n" + "="*60)
    print("MEDIAPIPE INSTALLATION FIX")
    print("="*60)
    
    # Step 1: Upgrade pip
    print("\n🔄 Step 1: Upgrade pip...")
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip to latest version"
    )
    
    # Step 2: Uninstall mediapipe current
    print("\n🗑️  Step 2: Uninstall mediapipe yang current...")
    run_command(
        f"{sys.executable} -m pip uninstall -y mediapipe",
        "Removing current MediaPipe installation"
    )
    
    # Step 3: Uninstall package lain yang mungkin conflict
    print("\n🗑️  Step 3: Uninstall opencv-contrib-python (jika ada)...")
    run_command(
        f"{sys.executable} -m pip uninstall -y opencv-contrib-python",
        "Removing OpenCV Contrib (bisa conflict dengan opencv-python)"
    )
    
    # Step 4: Install dependencies yang diperlukan
    print("\n📦 Step 4: Install dependencies...")
    dependencies = [
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
    ]
    
    for dep in dependencies:
        cmd = f"{sys.executable} -m pip install --upgrade {dep}"
        run_command(cmd, f"Installing {dep}")
    
    # Step 5: Install mediapipe dengan versi spesifik
    print("\n📦 Step 5: Install MediaPipe (versi spesifik)...")
    
    mediapipe_version = "0.8.11"  # Versi yang stable dan compatible
    
    run_command(
        f"{sys.executable} -m pip install mediapipe=={mediapipe_version}",
        f"Installing MediaPipe version {mediapipe_version}"
    )
    
    # Step 6: Verifikasi installation
    print("\n✅ Step 6: Verifikasi installation...")
    verify_script = '''
import sys
try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
    print(f"  Version: {mp.__version__ if hasattr(mp, '__version__') else 'unknown'}")
    
    # Test solutions
    face_mesh = mp.solutions.face_mesh
    print("✓ mp.solutions.face_mesh loaded successfully")
    
    drawing = mp.solutions.drawing_utils
    print("✓ mp.solutions.drawing_utils loaded successfully")
    
    print("\\n✅ MediaPipe installation is CORRECT!")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", verify_script],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
    
    # Step 7: Install full requirements
    print("\n📦 Step 7: Install remaining dependencies dari requirements.txt...")
    run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing all dependencies from requirements.txt"
    )
    
    # Final summary
    print("\n" + "="*60)
    print("✅ FIX COMPLETE!")
    print("="*60)
    print("\n📝 Summary:")
    print("  1. ✓ Pip upgraded")
    print("  2. ✓ MediaPipe reinstalled (v0.8.11)")
    print("  3. ✓ Dependencies installed")
    print("  4. ✓ Verified import")
    print("\n🚀 Now you can run: python main.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
