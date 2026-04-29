python -m venv venv
call venv/scripts/activate

python.exe -m pip install --upgrade pip
pip install requests onnxruntime pygame
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118

cmd
