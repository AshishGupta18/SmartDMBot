# Smart DMBot

Instructions:

note - add .env file in backend

1. Start backend-
   `cd backend`
   `python3 -m venv venv`
   `source venv/bin/activate`
   `pip install flask google-generativeai faiss-cpu PyPDF2 numpy`
   or
   `pip install -r requirements.txt`
   `python backend.py`

   For backend exe -
   run - 'pyinstaller --onefile --add-data "train_data;train_data" --add-data ".env;." --name MyBackendApp backend.py'

   Install d2 for flowchart
      'curl -fsSL https://d2lang.com/install.sh | sh -s --' or  https://github.com/terrastruct/d2/releases

2. Start frontend-
   `cd frontend`
   `npm install`
   `npm start`

   For frontend exe :
   'npm run dist'

3. Enjoy your chatbot.
   -cd - SmartDMBot\frontend\backend-bin.
   -copy MyAppBackendApp.exe from  ..SmartDMBot\backend\dist to  ..SmartDMBot\frontend\backend-bin.
   -In ..SmartDMBot\frontend\backend-bin run '.\MyBackendApp.exe'. wait for starting backend.
   -Go to file ..SmartDMBot\frontend\release-builds install "SmartDMBot Setup 1.0.0.exe".

   You can enjoy smartDMbot 🥳.

