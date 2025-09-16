from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <html>
        <body style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); color: white; font-family: Arial; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
            <div style="text-align: center;">
                <h1>Flask is Working! 🎉</h1>
                <p>The server is running successfully.</p>
                <p>Now you can run the main app.py file.</p>
            </div>
        </body>
    </html>
    '''

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Flask Test Server Starting...")
    print("="*50)
    print("\nOpen your browser and go to:")
    print("  → http://localhost:5000")
    print("  → http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)