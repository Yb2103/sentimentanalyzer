# Network Access Guide - Sentiment Analysis App

## 🌐 Making Your App Accessible from Other Devices

This guide explains how to access your Sentiment Analysis application from other devices on the same network (phones, tablets, other computers).

## 🚀 Quick Start (3 Steps)

### Step 1: Run the Admin Setup (One-time setup)
1. **Right-click** on `setup_and_run_admin.bat`
2. Select **"Run as Administrator"**
3. Follow the prompts - this configures your firewall

### Step 2: Get Your Network Address
After running the admin setup, you'll see something like:
```
Network access: http://172.20.171.84:5000
```
**This is your network address!**

### Step 3: Connect from Other Devices
- **Phone/Tablet**: Connect to same WiFi, open browser, go to `http://172.20.171.84:5000`
- **Other Computers**: Ensure same network, open browser, go to `http://172.20.171.84:5000`

---

## 📱 Device-Specific Instructions

### Mobile Phones & Tablets
1. Connect your phone/tablet to the **same WiFi network** as your computer
2. Open any web browser (Chrome, Safari, Firefox, etc.)
3. Type the network address: `http://[YOUR_IP]:5000`
4. The app should load normally

### Other Computers
1. Ensure the computer is on the **same local network**
2. Open any web browser
3. Navigate to: `http://[YOUR_IP]:5000`
4. Use the application as normal

### Smart TVs with Browsers
1. Connect TV to same WiFi network
2. Open TV's web browser
3. Navigate to the network address
4. Use TV remote to interact with the app

---

## 🔧 Available Startup Scripts

### For Network Access:
- **`setup_and_run_admin.bat`** ← **Recommended for first-time setup**
  - Configures firewall automatically
  - Shows network IP address
  - Starts the application

- **`start_sentiment_app_network.bat`** ← **For daily use**
  - Shows network access information
  - Doesn't need admin privileges after initial setup

### For Local-Only Access:
- **`RunThisWheneverYouNeedToRunTheProject.bat`** ← **Original script**
  - Only works on this computer
  - Use `localhost:5000`

---

## 🛠 Troubleshooting

### Problem: "Can't connect" or "Site can't be reached"

**Solution 1: Check Network Connection**
- Ensure all devices are on the same WiFi network
- Check your computer's IP address: `ipconfig`
- Verify the IP hasn't changed (WiFi networks sometimes reassign IPs)

**Solution 2: Firewall Issues**
```bash
# Run as Administrator:
netsh advfirewall firewall delete rule name="Sentiment Analysis App"
netsh advfirewall firewall add rule name="Sentiment Analysis App" dir=in action=allow protocol=TCP localport=5000
```

**Solution 3: Antivirus/Security Software**
- Temporarily disable antivirus
- Add Python.exe and your project folder to antivirus exceptions
- Some security software blocks network connections

**Solution 4: Port Already in Use**
```bash
# Check what's using port 5000:
netstat -ano | findstr :5000
```

### Problem: IP Address Changes

Your IP address might change when you:
- Restart your computer
- Disconnect/reconnect to WiFi  
- Connect to a different network

**Solution:** Re-run `setup_and_run_admin.bat` to get the new IP address

### Problem: App Loads But Features Don't Work

This usually means:
- **File uploads fail**: Check upload folder permissions
- **Analysis doesn't work**: Dependencies missing, run `pip install -r requirements.txt`
- **Slow performance**: Large files + weak network connection

---

## 🏗 Network Architecture

```
Your Computer (172.20.171.84:5000)
    ↓
WiFi Router/Network (172.20.168.1)
    ↓
Other Devices:
    📱 Phone (172.20.171.85)
    💻 Laptop (172.20.171.86)
    📺 Smart TV (172.20.171.87)
```

All devices need to be on the **same network segment** (same WiFi/LAN).

---

## 🔒 Security Considerations

### Current Security Level: **Development/Local Network**
- ✅ Safe for same WiFi network (home, office)
- ✅ Safe for trusted devices
- ❌ **NOT** safe for public internet exposure

### For Production Use:
- Add user authentication
- Use HTTPS instead of HTTP
- Implement API rate limiting
- Add input validation/sanitization

---

## ⚙ Configuration Files

### Flask Configuration (`config.py`):
```python
# Already configured for network access:
host='0.0.0.0'  # Accepts connections from any IP
port=5000       # Standard Flask port
```

### Firewall Rule:
```bash
Rule Name: "Sentiment Analysis App"
Direction: Inbound
Action: Allow  
Protocol: TCP
Local Port: 5000
```

---

## 📊 Usage Examples

### API Access from Other Devices:
```bash
# From another computer/device on the network:
curl -X POST http://172.20.171.84:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Great app!", "Needs improvement"]}'
```

### File Upload from Mobile:
1. Open browser on mobile: `http://172.20.171.84:5000`
2. Click "Choose File" 
3. Select CSV/Excel file from phone storage
4. Upload and analyze normally

---

## 🆘 Quick Help Commands

### Get Current IP Address:
```bash
ipconfig | findstr "IPv4 Address"
```

### Check If Port 5000 is Open:
```bash
netstat -an | findstr :5000
```

### Test Connection from Another Device:
```bash
# Replace IP with your actual IP:
ping 172.20.171.84
telnet 172.20.171.84 5000
```

### Reset Firewall Rule:
```bash
netsh advfirewall firewall delete rule name="Sentiment Analysis App"
netsh advfirewall firewall add rule name="Sentiment Analysis App" dir=in action=allow protocol=TCP localport=5000
```

---

## 📞 Support

If you're still having issues:

1. **Check the basics:**
   - Same WiFi network ✓
   - Firewall configured ✓  
   - Correct IP address ✓
   - App is running ✓

2. **Advanced debugging:**
   - Check Windows Event Viewer
   - Try different port (change in `config.py`)
   - Use network scanning tools

3. **Alternative solutions:**
   - Use mobile hotspot and connect both devices
   - Try USB tethering for direct connection
   - Set up VPN for remote access

---

**Made with ❤ for seamless sentiment analysis across all your devices!**