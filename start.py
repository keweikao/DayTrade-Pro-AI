#!/usr/bin/env python3
"""
Zeabur部署啟動腳本
"""
import os
import subprocess
import sys

def main():
    """主啟動函數"""
    
    # 設定環境變數
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'false')
    
    # 取得端口號
    port = os.environ.get('PORT', '8080')
    
    # 啟動Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    print(f"⚡ Starting DayTrade Pro AI System on port {port}")
    print(f"Command: {' '.join(cmd)}")
    
    # 執行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("👋 Application stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()