# snippet you can drop into a small test file
import numpy as np, torch, time
from run_trt import TRTUNet  # your class

x = torch.randn(1, 8, 32, 32, device="cuda", dtype=torch.float32)
t = torch.tensor([123], device="cuda", dtype=torch.int64)  # any valid timestep

fp32 = TRTUNet("/home/nazmus/Desktop/EdgeDiff SR/image-upscaler/src/unet.trt")
fp16 = TRTUNet("/home/nazmus/Desktop/EdgeDiff SR/image-upscaler/src/unet_fp16.trt")

# warmup
for _ in range(10):
    _ = fp32.infer(x, t); _ = fp16.infer(x, t)

torch.cuda.synchronize(); t0=time.perf_counter()
o32 = fp32.infer(x, t)
torch.cuda.synchronize(); t1=time.perf_counter()

torch.cuda.synchronize(); t2=time.perf_counter()
o16 = fp16.infer(x, t)
torch.cuda.synchronize(); t3=time.perf_counter()

o32_np = o32.detach().cpu().numpy()
o16_np = o16.detach().cpu().numpy()

print("FP32 ms:", (t1-t0)*1000, " FP16 ms:", (t3-t2)*1000)
print("max abs diff:", np.max(np.abs(o32_np - o16_np)))
print("mean abs diff:", np.mean(np.abs(o32_np - o16_np)))
