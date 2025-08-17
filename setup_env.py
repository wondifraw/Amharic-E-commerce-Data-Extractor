#!/usr/bin/env python3
"""Setup script to initialize .env file and validate configuration."""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Setup .env file from template."""
    project_root = Path(__file__).parent
    env_file = project_root / '.env'
    env_example = project_root / '.env.example'
    
    if not env_file.exists() and env_example.exists():
        print("📋 Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print("✅ .env file created!")
        print("⚠️  Please edit .env file with your actual credentials")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("❌ .env.example not found")
        return False
    
    # Validate configuration
    try:
        from src.config.env_config import config
        config.validate()
        print("✅ Configuration validated successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        print("💡 Please check your .env file")
        return False

if __name__ == "__main__":
    setup_environment()