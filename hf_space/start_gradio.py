"""
启动 Gradio 应用并生成公网分享链接
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import app

print("=" * 60)
print("Starting Gradio with share=True ...")
print("=" * 60)
app.demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
