"""
WSGI Entry Point for Render Deployment
"""
from app import app

# This is the WSGI application that gunicorn will use
application = app

if __name__ == "__main__":
    app.run()