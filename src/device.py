from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        test = torch.ones(1, device="cuda")
        _ = (test + 1).cpu()
        return torch.device("cuda")
    except RuntimeError as exc:
        print(f"CUDA is available but failed a runtime check; falling back to CPU. Error: {exc}")
        return torch.device("cpu")
