"""
Install additional dependencies for LSTM functionality
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install required packages for LSTM functionality"""
    packages = [
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0"
    ]
    
    print("Installing LSTM dependencies...")
    print("=" * 50)
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("=" * 50)
    print(f"Installation complete: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("✓ All dependencies installed successfully!")
        print("You can now use the LSTM functionality.")
    else:
        print("⚠ Some packages failed to install. Please install them manually.")

if __name__ == "__main__":
    main()