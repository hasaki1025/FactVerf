

# LORA在ReMote数据集上进行微调

当前项目基于Remote数据集对Qwen 3.5 9B在多模态实体-关系提取数据集上进行微调，训练VLLM在多模态信息上同时提取文本实体、图像实体、多模态关系任务

Remote数据集文章可通过PDF-Reader阅读lora/remote_lora/remote.pdf了解

## 数据预处理：Remote数据集
remote数据集为实体-关系提取数据集，提供了多模态信息的文本实体、图像实体、文本-图像实体关系、文本-文本实体关系、图像-图像实体关系，以下为Remote数据格式：
```json
{
    "id": 唯一标识符,
    "text": 原始文本,
    "image_id": 原始图像文件名,
    "entity": [
        {
            "type": "text", # 实体类型
            "text": "Pablo Casado", # 文本实体名称
            "pos": [] # 文本实体该字段无意义
        },
        {
            "type": "image", # 图像实体类型
            "text": "Unable to identify the image type.", # 图像实体文本描述
            "pos": [ # 图像实体位置（Box 坐标以及文件名）
                0.265,
                0.41,
                0.595,
                0.99,
                "6ef39b2e-bbe9-5c6d-9a00-4e01fa2312d2.jpg"
            ]
        }
    ],
    "rel": [ # 关系
        {
            "head": 0,
            "tail": 1,
            "relation": "none"
        }
    ]
}
```

- 图像文件可通过图像文件夹+图像文件名获取
- create_remote_dataset将加载数据集并将其转换成HF Dataset,具体写法可参考lora/DOCRED-FE_LORA/preprocess_dataset.py
- 数据处理流程如下：
  - 读取原始数据集
  - 将输入（原始文本和图像）拼成message列表
    ```json
    [
    {
        "role": "system",
        "content": [
            {
                "type": "text", 
                "text": system prompt
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": user_prompt_template + text
            },
           {
                "type": "image",
                # 直接填入本地路径
                "image": local_image_path 
            }
        ]
    }
    ]
    ```
  - 构建gt, 按照prompt中规定的输出格式构造gt message
  - 仿照lora/DOCRED-FE_LORA/preprocess_dataset.py中apply_prompt_template计算mask和label等值，并最终转换成hf dataset
  


