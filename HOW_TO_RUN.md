# How to Run the Sentiment Analysis Application

## 🚀 Quick Start

### Option 1: Network Access (Recommended) 🌐
**For accessing from phones, tablets, and other devices:**
1. **Right-click** `setup_and_run_admin.bat` and select **"Run as Administrator"**
2. Follow the prompts to configure firewall and get your network IP
3. Access from any device using the network IP (e.g., `http://172.20.171.84:5000`)

### Option 2: Daily Use (After Initial Setup)
1. Double-click **`start_sentiment_app_network.bat`**
2. This shows network access information without needing admin rights

### Option 3: Local Only (This Computer Only)
1. Double-click **`RunThisWheneverYouNeedToRunTheProject.bat`**
2. Access at: `http://localhost:5000`

### Option 4: Command Line
1. Open PowerShell or Command Prompt
2. Navigate to this directory:
   ```
   cd "C:\Users\Admin\Desktop\Projects\Sentiment analysis of comments received through E-consultation module"
   ```
3. Run:
   ```
   python app.py
   ```

## Access the Application

### Local Access (This Computer Only):
- **http://localhost:5000**
- **http://127.0.0.1:5000**

### Network Access (Other Devices):
- **http://[YOUR_IP_ADDRESS]:5000** (e.g., `http://172.20.171.84:5000`)
- Get your IP by running the admin setup script or using `ipconfig`

### Mobile/Tablet Access:
1. Connect your device to the **same WiFi network**
2. Open any web browser
3. Navigate to: `http://[YOUR_IP_ADDRESS]:5000`
4. The app works normally on mobile devices!

## Features

1. **Live Text Analysis**: Type text and get instant sentiment analysis
2. **File Upload**: Analyze CSV, Excel, JSON, or TXT files
3. **Demo Mode**: Try with sample data
4. **API Access**: Integrate with other applications

## Stop the Server

Press **Ctrl+C** in the command window to stop the server

## Project Structure

```
├── src/                    # Source code modules
│   ├── sentiment_analyzer.py
│   ├── text_summarizer.py
│   ├── wordcloud_generator.py
│   └── data_processor.py
├── templates/              # HTML templates
│   └── index.html
├── static/                 # CSS and JavaScript files
├── data/                   # Sample data
├── uploads/                # Uploaded files (temporary)
├── results/                # Analysis results
├── app.py                  # Main application (full features)
├── app_simple.py           # Simplified version (recommended)
├── app_realtime.py         # Real-time version with WebSocket
└── requirements.txt        # Python dependencies
```

## Troubleshooting

### If the application doesn't start:
1. Make sure Python is installed: `python --version`
2. Install dependencies: `pip install -r requirements.txt`
3. Check if port 5000 is free: `netstat -an | findstr :5000`

### If you see import errors:
Run: `pip install Flask pandas numpy nltk vaderSentiment textblob wordcloud matplotlib beautifulsoup4 openpyxl scikit-learn`

## Support

For issues or questions, refer to the README.md file in this directory.