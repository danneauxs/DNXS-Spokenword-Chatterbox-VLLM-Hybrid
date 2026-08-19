"""
Real TTS Performance Optimizer
Applies optimizations directly to ChatterboxTTS inference methods

This targets the actual bottlenecks:
1. T3 text-to-speech token generation
2. S3Gen speech token to audio generation
"""

import torch
import logging
from contextlib import contextmanager


class RealTTSOptimizer:
    """Real optimizations that actually affect TTS performance"""

    def __init__(self):
        """Context manager to temporarily disable mixed precision for a single operation.
        Args:
        None
        Returns:
        None
        """
        self.logger = logging.getLogger(__name__)
        self.original_methods = {}
        self.optimizations_applied = False
        self.fp32_fallback = False  # Flag to disable mixed precision for retries

    @contextmanager
    def fp32_fallback_mode(self):
        """Context manager to temporarily disable mixed precision for a single operation."""
        self.fp32_fallback = True
        print("⚠️ FP32 FALLBACK MODE ACTIVATED for problematic chunk.")
        try:
            yield
        finally:
            self.fp32_fallback = False
            print("✅ FP32 FALLBACK MODE DEACTIVATED.")

    def apply_optimizations(self, model):
        """Apply real optimizations to ChatterboxTTS model with detailed tracking"""
        optimizations_count = 0
        optimization_details = []

        print("🔍 DETAILED OPTIMIZATION ANALYSIS:")
        print(f"   Model type: {type(model)}")
        print(
            f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}"
        )

        try:
            # 1. Enable mixed precision for T3 inference
            if hasattr(model, "t3"):
                print(f"   T3 model found: {type(model.t3)}")
                if hasattr(model.t3, "inference"):
                    print(f"   T3.inference method found: {type(model.t3.inference)}")
                    self._optimize_t3_inference(model.t3)
                    optimizations_count += 1
                    optimization_details.append("T3 mixed precision")
                    print("✅ T3 inference optimized with mixed precision")
                else:
                    print("⚠️ T3 model has no inference method")
            else:
                print("⚠️ Model has no t3 attribute")

            # 2. Enable mixed precision for S3Gen inference
            if hasattr(model, "s3gen"):
                print(f"   S3Gen model found: {type(model.s3gen)}")
                if hasattr(model.s3gen, "inference"):
                    print(
                        f"   S3Gen.inference method found: {type(model.s3gen.inference)}"
                    )
                    self._optimize_s3gen_inference(model.s3gen)
                    optimizations_count += 1
                    optimization_details.append("S3Gen mixed precision")
                    print("✅ S3Gen inference optimized with mixed precision")
                else:
                    print("⚠️ S3Gen model has no inference method")
            else:
                print("⚠️ Model has no s3gen attribute")

            # 3. Enable torch.compile if available
            from config.config import ENABLE_TORCH_COMPILE

            if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
                torch_version = torch.__version__
                print(f"   PyTorch version: {torch_version}")
                try:
                    self._apply_torch_compile(model)
                    optimizations_count += 1
                    optimization_details.append("Torch.compile")
                    print("✅ Torch.compile applied to inference methods")
                except Exception as e:
                    print(f"⚠️ Torch.compile failed: {e}")
            else:
                print("⚠️ Torch.compile not available")

            # 4. Set optimal CUDA settings
            cuda_available = torch.cuda.is_available()
            print(f"   CUDA available: {cuda_available}")
            if cuda_available:
                device_name = torch.cuda.get_device_name()
                print(f"   CUDA device: {device_name}")

                self._optimize_cuda_settings()
                optimizations_count += 1
                optimization_details.append("CUDA settings")
                print("✅ CUDA settings optimized")
            else:
                print("⚠️ CUDA not available")

            self.optimizations_applied = True
            print("=" * 60)
            print("🎯 OPTIMIZATION SUMMARY:")
            print(f"   Applied optimizations: {optimizations_count}")
            print(f"   Details: {', '.join(optimization_details)}")
            print("=" * 60)

            return optimizations_count

        except Exception as e:
            print(f"❌ Real optimization failed: {e}")
            import traceback

            print(f"❌ Full traceback: {traceback.format_exc()}")
            return 0

    def _optimize_t3_inference(self, t3_model):
        """Optimize T3 text-to-speech inference"""
        if hasattr(t3_model, "inference"):
            self.original_methods["t3_inference"] = t3_model.inference

            def optimized_t3_inference(*args, **kwargs):
                """Optimizes S3Gen speech generation inference using mixed precision if enabled.
                Args:
                s3gen_model (object): The model to be optimized.
                Returns:
                None
                """
                if self.fp32_fallback:
                    return self.original_methods["t3_inference"](*args, **kwargs)
                else:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        return self.original_methods["t3_inference"](*args, **kwargs)

            t3_model.inference = optimized_t3_inference

    def _optimize_s3gen_inference(self, s3gen_model):
        """Optimize S3Gen speech generation inference"""
        if hasattr(s3gen_model, "inference"):
            self.original_methods["s3gen_inference"] = s3gen_model.inference

            def optimized_s3gen_inference(*args, **kwargs):
                """Apply torch.compile to inference methods Args: model (torch.nn.Module): The PyTorch model to compile Returns: None"""
                if self.fp32_fallback:
                    return self.original_methods["s3gen_inference"](*args, **kwargs)
                else:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        return self.original_methods["s3gen_inference"](*args, **kwargs)

            s3gen_model.inference = optimized_s3gen_inference

    def _apply_torch_compile(self, model):
        """Apply torch.compile to inference methods"""
        try:
            if hasattr(model, "t3") and hasattr(model.t3, "forward"):
                model.t3.forward = torch.compile(
                    model.t3.forward, mode="reduce-overhead"
                )

            if hasattr(model, "s3gen") and hasattr(model.s3gen, "forward"):
                model.s3gen.forward = torch.compile(
                    model.s3gen.forward, mode="reduce-overhead"
                )

        except Exception as e:
            print(f"⚠️ Torch.compile failed: {e}")

    def _optimize_cuda_settings(self):
        """Set optimal CUDA settings for inference"""
        if torch.cuda.is_available():
            # Enable TF32 for faster matrix operations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable CUDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True

            # Enable CUDNN deterministic mode for reproducibility
            torch.backends.cudnn.deterministic = False  # Faster but less deterministic

    def restore_original_methods(self, model):
        """Restore original methods (cleanup)"""
        if not self.optimizations_applied:
            return

        try:
            if "t3_inference" in self.original_methods and hasattr(model, "t3"):
                model.t3.inference = self.original_methods["t3_inference"]

            if "s3gen_inference" in self.original_methods and hasattr(model, "s3gen"):
                model.s3gen.inference = self.original_methods["s3gen_inference"]

            print("✅ Original inference methods restored")

        except Exception as e:
            print(f"❌ Failed to restore original methods: {e}")


# Global optimizer instance
_tts_optimizer = None


def get_tts_optimizer():
    """Get or create the global TTS optimizer"""
    global _tts_optimizer
    if _tts_optimizer is None:
        _tts_optimizer = RealTTSOptimizer()
    return _tts_optimizer


def optimize_chatterbox_model(model):
    """Apply real optimizations to ChatterboxTTS model"""
    optimizer = get_tts_optimizer()
    return optimizer.apply_optimizations(model)


@contextmanager
def optimized_inference(model):
    """Context manager for optimized inference"""
    optimizer = get_tts_optimizer()
    try:
        optimization_count = optimizer.apply_optimizations(model)
        print(f"🚀 Temporary optimizations applied: {optimization_count}")
        yield model
    finally:
        optimizer.restore_original_methods(model)
        print("🔄 Optimizations cleaned up")
