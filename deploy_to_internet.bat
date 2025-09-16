@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo     SENTIMENT ANALYSIS APP - PERMANENT INTERNET DEPLOYMENT
echo ========================================================================
echo.
echo This script will deploy your app to the internet PERMANENTLY with your
echo own custom domain name!
echo.
echo ========================================================================

:ask_domain
echo.
echo STEP 1: CHOOSE YOUR DOMAIN NAME
echo --------------------------------
echo.
echo Enter your desired domain/subdomain name:
echo (Examples: sentiment-analyzer, my-app, e-consultation)
echo.
set /p DOMAIN_NAME=Your domain name: 

if "!DOMAIN_NAME!"=="" (
    echo.
    echo ❌ Domain name cannot be empty!
    goto :ask_domain
)

echo.
echo Your app will be available at one of these URLs:
echo.
echo   Option 1 (Render):     https://!DOMAIN_NAME!.onrender.com
echo   Option 2 (Railway):    https://!DOMAIN_NAME!.up.railway.app  
echo   Option 3 (Vercel):     https://!DOMAIN_NAME!.vercel.app
echo   Option 4 (Netlify):    https://!DOMAIN_NAME!.netlify.app
echo.
echo Is this domain name correct? (Y/N)
set /p CONFIRM=Confirm: 

if /i not "!CONFIRM!"=="Y" goto :ask_domain

:choose_platform
echo.
echo ========================================================================
echo STEP 2: CHOOSE DEPLOYMENT PLATFORM
echo ========================================================================
echo.
echo Select your preferred hosting platform (ALL ARE FREE):
echo.
echo   [1] Render.com     - ✅ Best for Flask apps, Free forever
echo   [2] Railway.app    - ✅ Easy deployment, $5 free credit/month
echo   [3] Vercel         - ✅ Fast globally, Free forever
echo   [4] PythonAnywhere - ✅ Python-specific, Free tier available
echo   [5] Replit         - ✅ Online IDE + hosting, Free tier
echo.
set /p PLATFORM=Choose platform (1-5): 

if "!PLATFORM!"=="1" goto :deploy_render
if "!PLATFORM!"=="2" goto :deploy_railway
if "!PLATFORM!"=="3" goto :deploy_vercel
if "!PLATFORM!"=="4" goto :deploy_pythonanywhere
if "!PLATFORM!"=="5" goto :deploy_replit

echo Invalid choice. Please select 1-5.
goto :choose_platform

:deploy_render
echo.
echo ========================================================================
echo DEPLOYING TO RENDER.COM
echo ========================================================================
echo.
echo Your app will be available at: https://!DOMAIN_NAME!.onrender.com
echo.

REM Create deployment configuration with custom domain
echo services: > render_custom.yaml
echo   - type: web >> render_custom.yaml
echo     name: !DOMAIN_NAME! >> render_custom.yaml
echo     runtime: python >> render_custom.yaml
echo     plan: free >> render_custom.yaml
echo     buildCommand: "pip install -r requirements.txt" >> render_custom.yaml
echo     startCommand: "gunicorn app_production:app" >> render_custom.yaml
echo     envVars: >> render_custom.yaml
echo       - key: PYTHON_VERSION >> render_custom.yaml
echo         value: 3.11.0 >> render_custom.yaml
echo       - key: SECRET_KEY >> render_custom.yaml
echo         generateValue: true >> render_custom.yaml
echo     autoDeploy: true >> render_custom.yaml

echo.
echo ✅ Configuration created: render_custom.yaml
echo.
echo NEXT STEPS:
echo -----------
echo 1. Create account at: https://render.com/signup
echo 2. Install Git if not installed: https://git-scm.com/download/win
echo 3. Initialize Git repository:
echo.
echo    git init
echo    git add .
echo    git commit -m "Initial deployment"
echo.
echo 4. Push to GitHub:
echo    - Create new repository at: https://github.com/new
echo    - Name it: !DOMAIN_NAME!
echo    - Follow GitHub's instructions to push
echo.
echo 5. Deploy on Render:
echo    - Go to: https://dashboard.render.com/new/web
echo    - Connect GitHub repository
echo    - Name: !DOMAIN_NAME!
echo    - Click "Create Web Service"
echo.
echo 6. Your app will be LIVE at: https://!DOMAIN_NAME!.onrender.com
echo.
pause
goto :create_github_files

:deploy_railway
echo.
echo ========================================================================
echo DEPLOYING TO RAILWAY.APP
echo ========================================================================
echo.
echo Your app will be available at: https://!DOMAIN_NAME!.up.railway.app
echo.

REM Create Railway configuration
echo { > railway.json
echo   "name": "!DOMAIN_NAME!", >> railway.json
echo   "description": "Sentiment Analysis Web Application", >> railway.json
echo   "repository": "https://github.com/yourusername/!DOMAIN_NAME!", >> railway.json
echo   "env": { >> railway.json
echo     "PORT": 5000, >> railway.json
echo     "SECRET_KEY": "${{SECRET_KEY}}" >> railway.json
echo   } >> railway.json
echo } >> railway.json

echo.
echo ✅ Configuration created: railway.json
echo.
echo NEXT STEPS:
echo -----------
echo 1. Create account at: https://railway.app/signup
echo 2. Install Railway CLI:
echo    npm install -g @railway/cli
echo.
echo 3. Deploy with one command:
echo    railway login
echo    railway init -n !DOMAIN_NAME!
echo    railway up
echo.
echo 4. Your app will be LIVE at: https://!DOMAIN_NAME!.up.railway.app
echo.
pause
goto :create_github_files

:deploy_vercel
echo.
echo ========================================================================
echo DEPLOYING TO VERCEL
echo ========================================================================
echo.
echo Your app will be available at: https://!DOMAIN_NAME!.vercel.app
echo.

REM Create Vercel configuration
echo { > vercel.json
echo   "name": "!DOMAIN_NAME!", >> vercel.json
echo   "version": 2, >> vercel.json
echo   "builds": [ >> vercel.json
echo     { >> vercel.json
echo       "src": "app_production.py", >> vercel.json
echo       "use": "@vercel/python" >> vercel.json
echo     } >> vercel.json
echo   ], >> vercel.json
echo   "routes": [ >> vercel.json
echo     { >> vercel.json
echo       "src": "/(.*)", >> vercel.json
echo       "dest": "app_production.py" >> vercel.json
echo     } >> vercel.json
echo   ] >> vercel.json
echo } >> vercel.json

echo.
echo ✅ Configuration created: vercel.json
echo.
echo NEXT STEPS:
echo -----------
echo 1. Create account at: https://vercel.com/signup
echo 2. Install Vercel CLI:
echo    npm install -g vercel
echo.
echo 3. Deploy with one command:
echo    vercel --name !DOMAIN_NAME!
echo.
echo 4. Your app will be LIVE at: https://!DOMAIN_NAME!.vercel.app
echo.
pause
goto :create_github_files

:deploy_pythonanywhere
echo.
echo ========================================================================
echo DEPLOYING TO PYTHONANYWHERE
echo ========================================================================
echo.
echo Your app will be available at: https://!DOMAIN_NAME!.pythonanywhere.com
echo.

echo.
echo MANUAL STEPS FOR PYTHONANYWHERE:
echo ---------------------------------
echo 1. Sign up at: https://www.pythonanywhere.com/registration/register/beginner/
echo    Username: !DOMAIN_NAME!
echo.
echo 2. After login, go to "Web" tab
echo 3. Click "Add a new web app"
echo 4. Choose "Flask" and Python 3.10
echo 5. Upload your project files
echo 6. Configure as shown in the guide
echo.
echo Your app will be at: https://!DOMAIN_NAME!.pythonanywhere.com
echo.
pause
goto :create_github_files

:deploy_replit
echo.
echo ========================================================================
echo DEPLOYING TO REPLIT
echo ========================================================================
echo.
echo Your app will be available at: https://!DOMAIN_NAME!.repl.co
echo.

REM Create Replit configuration
echo run = "python app_production.py" > .replit
echo language = "python3" >> .replit
echo.
echo [env] >> .replit
echo SECRET_KEY = "your-secret-key-here" >> .replit

echo.
echo ✅ Configuration created: .replit
echo.
echo NEXT STEPS:
echo -----------
echo 1. Go to: https://replit.com/signup
echo 2. Click "Create Repl"
echo 3. Choose "Import from GitHub" or upload files
echo 4. Name: !DOMAIN_NAME!
echo 5. Click "Run" button
echo.
echo Your app will be at: https://!DOMAIN_NAME!.repl.co
echo.
pause
goto :create_github_files

:create_github_files
echo.
echo ========================================================================
echo CREATING GITHUB DEPLOYMENT FILES
echo ========================================================================

REM Create .gitignore
echo __pycache__/ > .gitignore
echo *.py[cod] >> .gitignore
echo *$py.class >> .gitignore
echo *.so >> .gitignore
echo .Python >> .gitignore
echo env/ >> .gitignore
echo venv/ >> .gitignore
echo uploads/ >> .gitignore
echo results/ >> .gitignore
echo *.log >> .gitignore
echo .env >> .gitignore
echo instance/ >> .gitignore
echo .webassets-cache >> .gitignore
echo .pytest_cache/ >> .gitignore

echo ✅ Created: .gitignore

REM Create runtime.txt for Python version
echo python-3.11.0 > runtime.txt
echo ✅ Created: runtime.txt

REM Update requirements.txt for production
echo gunicorn==21.2.0 >> requirements.txt
echo ✅ Updated: requirements.txt with gunicorn

REM Create environment file template
echo SECRET_KEY=change-this-to-random-string > .env.example
echo PORT=5000 >> .env.example
echo DEBUG=False >> .env.example
echo ✅ Created: .env.example

echo.
echo ========================================================================
echo           🎉 DEPLOYMENT PREPARATION COMPLETE! 🎉
echo ========================================================================
echo.
echo DOMAIN NAME: !DOMAIN_NAME!
echo.
echo Your app will be permanently available at your chosen URL!
echo.
echo IMPORTANT FILES CREATED:
echo   ✅ app_production.py - Production-ready app
echo   ✅ Deployment configs for your chosen platform
echo   ✅ .gitignore - Ignore unnecessary files
echo   ✅ runtime.txt - Python version specification
echo.
echo TO COMPLETE DEPLOYMENT:
echo   1. Choose and follow platform-specific steps above
echo   2. Your app will be LIVE in 5-10 minutes
echo   3. Share the URL with anyone in the world!
echo.
echo ========================================================================
echo.
pause