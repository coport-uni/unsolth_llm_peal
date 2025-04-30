import subprocess
import os
import time

# Wait for a few seconds for Ollama to load!
subprocess.Popen(["ollama", "serve"])
time.sleep(4)
print("ollama running")

os.system("ollama create peal_llm -f ./model_test/Modelfile")
time.sleep(4)
print("model_loaded")

os.system("conda init")
os.system("conda activate ollama")
os.system("open-webui serve --port 6864")
print("GUI_loaded")