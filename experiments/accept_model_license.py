#!/usr/bin/env python3
# accept_model_license.py - 接受模型许可证并下载模型

import os
import requests
from huggingface_hub import HfApi, hf_hub_download

def accept_model_license():
    """接受Nemotron-3-8B-SteerLM模型许可证"""
    print("🔐 Accepting model license...")
    
    # 模型信息
    model_id = "nvidia/nemotron-3-8b-chat-4k-steerlm"
    
    try:
        # 创建API客户端
        api = HfApi()
        
        # 获取模型信息
        model_info = api.model_info(model_id)
        print(f"📋 Model: {model_info.modelId}")
        print(f"📝 License: {model_info.cardData.get('license', 'Unknown')}")
        
        # 尝试接受许可证
        try:
            api.accept_terms_of_use(model_id)
            print("✅ License accepted successfully!")
        except Exception as e:
            print(f"⚠️  License acceptance failed: {e}")
            print("💡 You may need to manually accept the license at:")
            print(f"   https://huggingface.co/{model_id}")
        
        # 尝试下载模型
        print("📥 Attempting to download model...")
        try:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename="Nemotron-3-8B-Chat-4k-SteerLM.nemo",
                cache_dir="/root/.cache/huggingface",
                resume_download=True
            )
            print(f"✅ Model downloaded successfully to: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """主函数"""
    print("🚀 Starting model license acceptance...")
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
    
    # 接受许可证并下载模型
    model_path = accept_model_license()
    
    if model_path:
        print(f"\n🎉 Success! Model ready at: {model_path}")
        print("💡 You can now run: python3 steerlm_nemo_fixed.py")
    else:
        print("\n❌ Failed to download model")
        print("💡 Please manually accept the license at:")
        print("   https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")

if __name__ == "__main__":
    main() 