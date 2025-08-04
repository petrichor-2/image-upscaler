import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch

class TRTUNet:
    def __init__(self, engine_path="unet.trt"):
        """
        Initialize the TensorRT engine for the UNet model.
        """
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context() # Create execution context for inference
        self.host_mem, self.device_mem, self.bindings, self.device_bindings, self.stream = self._allocate_buffers()

    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        bindings = {} # Dictionary to hold tensor names and their device memory addresses
        device_bindings = [] # List to hold device memory addresses for execution
        host_mem = {} # stores CPU-side memory for each input/output tensor
        device_mem = {} # stores GPU-side memory for each input/output tensor
        stream = cuda.Stream() # Create a CUDA stream for asynchronous execution

        for name in self.engine:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            # Create host memory for input/output tensors
            host_mem[name] = cuda.pagelocked_empty(size, dtype)
            # Allocate device memory for input/output tensors
            device_mem[name] = cuda.mem_alloc(host_mem[name].nbytes)
            # Records the device pointer for this tensor in bindings
            bindings[name] = int(device_mem[name])
            device_bindings.append(int(device_mem[name]))

        # returns CPU buffer, GPU buffer, bindings dict, device bindings list and CUDA stream
        return host_mem, device_mem, bindings, device_bindings, stream

    def infer(self, x, t, device):
        """
        Run inference with the UNet model.

        Args:
            x: np.ndarray of shape (1, 8, 32, 32), dtype float32
            t: np.ndarray of shape (1,), dtype int64 or int32

        Returns:
            np.ndarray of shape (1, 4, 32, 32), dtype float32
        """
        # Convert torch.Tensor to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()

        assert x.shape == (1, 8, 32, 32)
        assert t.shape == (1,)
        assert x.dtype == np.float32
        assert t.dtype in (np.int32, np.int64)

        # Copy inputs to host memory
        np.copyto(self.host_mem["x"], x.ravel())
        np.copyto(self.host_mem["timestep"], t.ravel())

        # Copy to device and bind
        for name in self.host_mem:
            cuda.memcpy_htod_async(self.device_mem[name], self.host_mem[name], self.stream)
            self.context.set_tensor_address(name, self.bindings[name])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output back
        cuda.memcpy_dtoh_async(self.host_mem["predicted_noise"], self.device_mem["predicted_noise"], self.stream)
        self.stream.synchronize()

        # Reshape, convert to tensor and return
        output_np = self.host_mem["predicted_noise"].reshape(1, 4, 32, 32)
        return torch.from_numpy(output_np).to(device)