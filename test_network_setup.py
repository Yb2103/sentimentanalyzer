#!/usr/bin/env python3
"""
Quick Network Test for Sentiment Analysis App
Tests network configuration and suggests next steps
"""

import socket
import subprocess
import sys
import re
from flask import Flask

def get_local_ip():
    """Get the local IP address"""
    try:
        # Create a socket to find the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None

def check_port_availability(port=5000):
    """Check if port 5000 is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True
    except OSError:
        return False

def check_firewall_rule():
    """Check if firewall rule exists"""
    try:
        result = subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name="Sentiment Analysis App"'],
            capture_output=True, text=True, shell=True
        )
        return "Sentiment Analysis App" in result.stdout
    except Exception:
        return False

def get_network_interfaces():
    """Get network interface information"""
    try:
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        
        # Extract IPv4 addresses
        ipv4_pattern = r'IPv4 Address.*:\s*([0-9.]+)'
        ipv4_addresses = re.findall(ipv4_pattern, result.stdout)
        
        # Filter out localhost
        network_ips = [ip for ip in ipv4_addresses if not ip.startswith('127.')]
        return network_ips
    except Exception:
        return []

def main():
    print("=" * 70)
    print("  SENTIMENT ANALYSIS APP - NETWORK CONFIGURATION TEST")
    print("=" * 70)
    print()
    
    # Test 1: Python and Flask
    print("✓ Python version:", sys.version.split()[0])
    try:
        import flask
        print("✓ Flask is available")
    except ImportError:
        print("❌ Flask not installed! Run: pip install flask")
        return
    
    print()
    
    # Test 2: Network Configuration
    print("📡 NETWORK CONFIGURATION:")
    print("-" * 30)
    
    local_ip = get_local_ip()
    network_ips = get_network_interfaces()
    
    print(f"Primary IP Address: {local_ip or 'Unable to determine'}")
    print("All Network IPs:", network_ips if network_ips else ["None found"])
    
    if local_ip:
        print(f"\n🌐 Network Access URL: http://{local_ip}:5000")
    
    print()
    
    # Test 3: Port Availability
    print("🔌 PORT CONFIGURATION:")
    print("-" * 30)
    port_available = check_port_availability()
    if port_available:
        print("✓ Port 5000 is available")
    else:
        print("❌ Port 5000 is in use")
        print("   Try: netstat -ano | findstr :5000")
    
    print()
    
    # Test 4: Firewall
    print("🔥 FIREWALL CONFIGURATION:")
    print("-" * 30)
    firewall_configured = check_firewall_rule()
    if firewall_configured:
        print("✓ Firewall rule exists")
    else:
        print("❌ Firewall rule not found")
        print("   Need to run: setup_and_run_admin.bat as Administrator")
    
    print()
    
    # Test 5: Quick Flask Test
    print("🧪 QUICK FLASK TEST:")
    print("-" * 30)
    try:
        app = Flask(__name__)
        
        @app.route('/test')
        def test():
            return "Network test successful!"
        
        print("✓ Flask app can be created")
        
        # Test if we can bind to all interfaces
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind(('0.0.0.0', 0))  # Bind to any available port
            test_socket.close()
            print("✓ Can bind to all network interfaces (0.0.0.0)")
        except Exception as e:
            print(f"❌ Cannot bind to all interfaces: {e}")
            
    except Exception as e:
        print(f"❌ Flask test failed: {e}")
    
    print()
    
    # Summary and Next Steps
    print("📋 SUMMARY & NEXT STEPS:")
    print("=" * 30)
    
    if local_ip and port_available:
        print("✅ READY FOR NETWORK ACCESS!")
        print(f"   Your app will be accessible at: http://{local_ip}:5000")
        print()
        
        if not firewall_configured:
            print("⚠️  FIREWALL SETUP NEEDED:")
            print("   1. Right-click 'setup_and_run_admin.bat'")
            print("   2. Select 'Run as Administrator'")
            print("   3. This will configure Windows Firewall")
            print()
        
        print("🚀 TO START THE APP:")
        print("   - For network access: Double-click 'start_sentiment_app_network.bat'")
        print("   - For admin setup: Right-click 'setup_and_run_admin.bat' → Run as Admin")
        print()
        
        print("📱 TO ACCESS FROM OTHER DEVICES:")
        print(f"   1. Connect device to same WiFi network")
        print(f"   2. Open browser and go to: http://{local_ip}:5000")
        print()
        
    else:
        print("❌ ISSUES DETECTED:")
        if not local_ip:
            print("   - Cannot determine network IP address")
            print("   - Check network connection")
        if not port_available:
            print("   - Port 5000 is already in use")
            print("   - Stop other applications using this port")
        print()
        print("🔧 TROUBLESHOOTING:")
        print("   - Restart your computer")
        print("   - Check WiFi connection")
        print("   - Run: netstat -ano | findstr :5000")
    
    print("=" * 70)

if __name__ == "__main__":
    main()