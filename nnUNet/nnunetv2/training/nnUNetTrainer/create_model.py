import torch



def create_model(module):  # 更清晰的参数名（避免命名冲突）
    print("=" * 20)
    print("now use our snake")
    print("=" * 20)
    return module()  # ✅ 调用传入的模块创建实例



