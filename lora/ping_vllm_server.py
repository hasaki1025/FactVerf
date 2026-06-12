import os
from openai import OpenAI

# ---------------------------------------------------------
# 配置区
# ---------------------------------------------------------
# 请替换为你的远程服务器 IP 或域名。注意路径通常需要以 /v1 结尾！
# 例如: "http://192.168.1.100:8000/v1" 或 "https://api.yourdomain.com/v1"
BASE_URL = "http://127.0.0.1:8001/v1"

# 如果是访问真实的 OpenAI 或商业中转站，填写真实 API Key。
# 如果是访问自己部署的 vLLM / FastChat 等，通常填 "EMPTY" 即可。
API_KEY = "EMPTY"


# 可选：如果你之前遇到过 502 网关错误，可以取消下面三行的注释以屏蔽本地代理
# os.environ["HTTP_PROXY"] = ""
# os.environ["HTTPS_PROXY"] = ""
# os.environ["ALL_PROXY"] = ""

def run_diagnostics():
    print(f"🚀 开始测试远程 OpenAI 兼容服务端: {BASE_URL}")
    print("-" * 50)

    # 1. 初始化客户端
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=30.0  # 设置 30 秒超时，防止死等
    )

    # 2. 测试第一步：获取模型列表
    print("[测试 1/2] 正在拉取可用模型列表 (/v1/models)...")
    try:
        models_response = client.models.list()
        # 提取所有模型的名称
        available_models = [model.id for model in models_response.data]

        if not available_models:
            print("⚠️ 警告：成功连接到服务器，但服务器返回的模型列表为空！")
            return

        print(f"✅ 成功！当前服务器共有 {len(available_models)} 个可用模型。")
        for idx, m_name in enumerate(available_models):
            print(f"   {idx + 1}. {m_name}")

        # 自动选择第一个模型用于下一步测试
        test_model_name = available_models[0]

    except Exception as e:
        print(f"❌ 失败：无法获取模型列表。请检查网络或 URL 是否正确。")
        print(f"具体报错: {e}")
        return

    print("-" * 50)

    # 3. 测试第二步：执行对话推理
    print(f"[测试 2/2] 正在使用模型 '{test_model_name}' 进行对话生成测试 (/v1/chat/completions)...")
    try:
        response = client.chat.completions.create(
            model=test_model_name,
            messages=[
                {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
                {"role": "user", "content": "你好！请用一句话做个简单的自我介绍。"}
            ],
            temperature=0.3,
            max_tokens=100
        )

        reply = response.choices[0].message.content
        print("✅ 成功！模型推理正常，返回结果如下：")
        print(f"🤖 模型回复: {reply}")

    except Exception as e:
        print(f"❌ 失败：文本生成请求被拒绝或处理异常。")
        print(f"具体报错: {e}")

    print("-" * 50)
    print("🎉 诊断结束。")


if __name__ == "__main__":
    run_diagnostics()