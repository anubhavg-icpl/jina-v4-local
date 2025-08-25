#!/usr/bin/env python3
"""
Setup script for Jina Embeddings v4 Hello World Project

This script helps users set up the project environment and run initial tests.

Author: Claude
Date: 2025
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_system():
    """Check system information"""
    system = platform.system()
    machine = platform.machine()
    
    print(f"💻 System: {system} {platform.release()}")
    print(f"🏗️  Architecture: {machine}")
    
    # Check for Apple Silicon
    if system == "Darwin" and machine == "arm64":
        print("🍎 Apple Silicon detected - MPS acceleration will be available!")
    elif system == "Darwin":
        print("🍎 Intel Mac detected - CPU only")
    else:
        print("🐧 Non-Mac system detected - CPU only")
    
    return True


def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing requirements...")
        
        # Determine pip path
        if platform.system() == "Windows":
            pip_path = "venv\\Scripts\\pip"
        else:
            pip_path = "venv/bin/pip"
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("💡 Try running manually:")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        return False


def run_hello_world_test():
    """Run a basic test of the hello world script"""
    try:
        print("🧪 Running hello world test...")
        
        # Determine python path
        if platform.system() == "Windows":
            python_path = "venv\\Scripts\\python"
        else:
            python_path = "venv/bin/python"
        
        # Run test with limited output
        result = subprocess.run(
            [python_path, "-c", """
import sys
sys.path.append('.')
from hello_world import JinaEmbeddingsV4
print('✅ Import successful')
"""], 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Basic import test passed")
            return True
        else:
            print("❌ Import test failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out - this might be due to model downloading")
        print("💡 Try running 'python hello_world.py' manually")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def create_activation_script():
    """Create easy activation script"""
    if platform.system() == "Windows":
        script_content = """@echo off
echo 🚀 Activating Jina Embeddings v4 Environment...
call venv\\Scripts\\activate.bat
echo ✅ Environment activated!
echo 💡 Run 'python hello_world.py' to get started
cmd /k
"""
        with open("activate.bat", "w") as f:
            f.write(script_content)
        print("📝 Created activation script: activate.bat")
    else:
        script_content = """#!/bin/bash
echo "🚀 Activating Jina Embeddings v4 Environment..."
source venv/bin/activate
echo "✅ Environment activated!"
echo "💡 Run 'python hello_world.py' to get started"
exec "$SHELL"
"""
        with open("activate.sh", "w") as f:
            f.write(script_content)
        os.chmod("activate.sh", 0o755)
        print("📝 Created activation script: activate.sh")


def main():
    """Main setup function"""
    print("=" * 60)
    print("🌟 JINA EMBEDDINGS V4 - PROJECT SETUP 🌟")
    print("=" * 60)
    
    # Run checks
    if not check_python_version():
        sys.exit(1)
    
    check_system()
    
    # Setup steps
    print("\n" + "=" * 40)
    print("📋 SETUP STEPS")
    print("=" * 40)
    
    success = True
    
    # Step 1: Create virtual environment
    if not create_virtual_environment():
        success = False
    
    # Step 2: Install requirements
    if success and not install_requirements():
        success = False
    
    # Step 3: Create activation script
    create_activation_script()
    
    # Step 4: Run basic test
    if success:
        print("\n🧪 Running basic compatibility test...")
        run_hello_world_test()
    
    # Final instructions
    print("\n" + "=" * 60)
    if success:
        print("🎉 SETUP COMPLETED SUCCESSFULLY! 🎉")
    else:
        print("⚠️  SETUP COMPLETED WITH ISSUES")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    if platform.system() == "Windows":
        print("   1. Run: activate.bat")
    else:
        print("   1. Run: source venv/bin/activate")
    print("   2. Run: python hello_world.py")
    print("   3. Explore: python examples/text_similarity.py")
    print("   4. Advanced: python examples/multimodal_search.py")
    
    print(f"\n📚 Documentation: README.md")
    print(f"🆘 Troubleshooting: Check README.md troubleshooting section")


if __name__ == "__main__":
    main()