import time,torch

def get_time():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.synchronize()
    # stTime = time.time()
    stTime = time.perf_counter()

    return stTime







