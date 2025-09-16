#!/usr/bin/env python3
"""
Test script to verify the app can start properly for Render deployment
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from app import app
        print("✅ App imported successfully")
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False
    
    try:
        from config import Config
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    try:
        from src import SentimentAnalyzer, DataProcessor
        print("✅ Core modules imported successfully")
    except Exception as e:
        print(f"❌ Failed to import core modules: {e}")
        return False
    
    return True

def test_app_creation():
    """Test if Flask app can be created"""
    print("\nTesting Flask app creation...")
    
    try:
        from app import app
        
        # Test basic routes
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Home route works")
            else:
                print(f"⚠️ Home route returned status {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nTesting required directories...")
    
    required_dirs = ['uploads', 'results', 'templates', 'static', 'src']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name} directory exists")
        else:
            print(f"⚠️ {dir_name} directory missing - will be created at runtime")
    
    return True

if __name__ == '__main__':
    print("Render Deployment Readiness Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_directories()
    success &= test_app_creation()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ All tests passed! App should deploy successfully.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the issues above.")
        sys.exit(1)