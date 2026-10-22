"""检查 PyTorch 能否识别 GPU，并在 GPU 上跑一次矩阵乘。"""

import torch
import transformers


def main() -> None:
    print(f"torch       : {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"capability  : sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"memory      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print(f"bf16 matmul : ok (mean abs = {y.float().abs().mean().item():.3f})")


if __name__ == "__main__":
    main()
