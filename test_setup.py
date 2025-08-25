#!/usr/bin/env python3
"""
Quick setup test for Jina Embeddings v4 Project

This script tests basic functionality without downloading the full model.

Author: Claude
Date: 2025
"""

import sys
import importlib.util
import platform
from pathlib import Path


def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Incompatible (need 3.8+)")
        return False


def test_file_structure():
    """Test if all required files exist"""
    print("📁 Testing file structure...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "hello_world.py",
        "config.py",
        "setup.py",
        "examples/text_similarity.py",
        "examples/multimodal_search.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    else:
        print(f"   ✅ All {len(required_files)} required files present")
        return True


def test_imports():
    """Test basic imports without loading heavy dependencies"""
    print("📦 Testing basic imports...")
    
    basic_imports = [
        "os", "sys", "pathlib", "typing", "time", "json"
    ]
    
    failed_imports = []
    for module_name in basic_imports:
        try:
            importlib.import_module(module_name)
        except ImportError:
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"   ❌ Failed imports: {failed_imports}")
        return False
    else:
        print(f"   ✅ All {len(basic_imports)} basic imports successful")
        return True


def test_file_syntax():
    """Test Python syntax of all project files"""
    print("🔍 Testing file syntax...")
    
    python_files = [
        "hello_world.py",
        "config.py", 
        "setup.py",
        "examples/text_similarity.py",
        "examples/multimodal_search.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append((file_path, str(e)))
        except FileNotFoundError:
            syntax_errors.append((file_path, "File not found"))
    
    if syntax_errors:
        print("   ❌ Syntax errors found:")
        for file_path, error in syntax_errors:
            print(f"      {file_path}: {error}")
        return False
    else:
        print(f"   ✅ All {len(python_files)} Python files have valid syntax")
        return True


def test_config():
    """Test configuration loading"""
    print("⚙️  Testing configuration...")
    
    try:
        import config
        config.Config.create_directories()
        device = config.Config.get_device()
        model_config = config.Config.get_model_config()
        
        print(f"   ✅ Configuration loaded successfully")
        print(f"   📱 Device: {device}")
        print(f"   🤖 Model: {model_config['model_name']}")
        return True
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False


def test_system_info():
    """Display system information"""
    print("💻 System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("   🍎 Apple Silicon detected - MPS acceleration available!")
    elif platform.system() == "Darwin":
        print("   🍎 Intel Mac detected - CPU only")
    else:
        print("   🐧 Non-Mac system - CPU only")


def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 JINA EMBEDDINGS V4 - SETUP TEST 🧪")
    print("=" * 60)
    
    test_system_info()
    
    print("\n" + "=" * 40)
    print("📋 RUNNING TESTS")
    print("=" * 40)
    
    tests = [
        ("Python Version", test_python_version),
        ("File Structure", test_file_structure),
        ("Basic Imports", test_imports),
        ("File Syntax", test_file_syntax),
        ("Configuration", test_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ Your setup is ready to go!")
        print("\n📋 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run hello world: python hello_world.py")
    else:
        print(f"⚠️  {passed}/{total} TESTS PASSED")
        print("❌ Some issues need to be resolved:")
        
        for test_name, result in results:
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()