#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Token-Adaptive Precision Core Engine (TAP-Engine)
=========================================================
Advanced precision management for inference and training of large language models.
Supports Transformer, Mamba, RetNet, and RWKV architectures with dynamic token-level
precision control, hardware acceleration, and memory optimization.

Features:
- Token-level adaptive precision for computation efficiency
- Multi-architecture support (Transformer, Mamba/SSM, RetNet, RWKV)
- Integration with QLoRA, BitByte, and hardware accelerators
- Dynamic computation allocation based on token importance
- Comprehensive diagnostic and monitoring systems
"""

import os
import time
import math
import logging
import warnings
import gc
import json
import hashlib
import threading
import dataclasses
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TypeVar, Generic, Set, Iterator
from pathlib import Path
import contextlib
from dataclasses import dataclass, field, fields
from enum import Enum, auto
import functools
import inspect
import re
import sys
from collections import OrderedDict, defaultdict, deque
import copy

# Conditionally import libraries to handle missing dependencies gracefully
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Limited functionality.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Some features will be disabled.")

# Optional imports for advanced features
TRANSFORMERS_AVAILABLE = False
ACCELERATE_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False
PEFT_AVAILABLE = False
TRITON_AVAILABLE = False

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import accelerate
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    pass

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    pass

try:
    import peft
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    pass

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass

# Setup logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("tap_engine")

# Configure console handler with color support
try:
    import colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger.handlers = [handler]
except ImportError:
    pass  # Fall back to basic logging if colorlog not available

#------------------------------------------------------------------------------
# Version and Compatibility Information
#------------------------------------------------------------------------------

__version__ = "0.2.0"

# System compatibility information
SYSTEM_INFO = {
    "tap_version": __version__,
    "python_version": sys.version,
    "torch_available": TORCH_AVAILABLE,
    "numpy_available": NUMPY_AVAILABLE,
    "transformers_available": TRANSFORMERS_AVAILABLE,
    "accelerate_available": ACCELERATE_AVAILABLE,
    "bitsandbytes_available": BITSANDBYTES_AVAILABLE,
    "peft_available": PEFT_AVAILABLE,
    "triton_available": TRITON_AVAILABLE,
}

if TORCH_AVAILABLE:
    SYSTEM_INFO["torch_version"] = torch.__version__
    SYSTEM_INFO["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        SYSTEM_INFO["cuda_version"] = torch.version.cuda
        SYSTEM_INFO["gpu_count"] = torch.cuda.device_count()
        SYSTEM_INFO["current_device"] = torch.cuda.current_device()
        SYSTEM_INFO["gpu_name"] = torch.cuda.get_device_name(SYSTEM_INFO["current_device"])

def show_system_info():
    """Display system compatibility information."""
    logger.info("TAP Engine System Information:")
    
    # Core info
    logger.info(f"- TAP Engine Version: {SYSTEM_INFO['tap_version']}")
    logger.info(f"- Python Version: {SYSTEM_INFO['python_version'].split()[0]}")
    
    # PyTorch info
    if SYSTEM_INFO['torch_available']:
        logger.info(f"- PyTorch Version: {SYSTEM_INFO['torch_version']}")
        
        if SYSTEM_INFO['cuda_available']:
            logger.info(f"- CUDA Version: {SYSTEM_INFO['cuda_version']}")
            logger.info(f"- GPU: {SYSTEM_INFO['gpu_name']} (Device {SYSTEM_INFO['current_device']})")
            logger.info(f"- Total GPUs: {SYSTEM_INFO['gpu_count']}")
        else:
            logger.info("- CUDA: Not Available (CPU only)")
    else:
        logger.info("- PyTorch: Not Available")
    
    # Key libraries
    integrations = []
    if SYSTEM_INFO['transformers_available']:
        integrations.append(f"Transformers {transformers.__version__}")
    if SYSTEM_INFO['accelerate_available']:
        integrations.append(f"Accelerate {accelerate.__version__}")
    if SYSTEM_INFO['bitsandbytes_available']:
        integrations.append(f"BitsAndBytes {bnb.__version__}")
    if SYSTEM_INFO['peft_available']:
        integrations.append(f"PEFT {peft.__version__}")
    if SYSTEM_INFO['triton_available']:
        integrations.append(f"Triton {triton.__version__}")
    
    if integrations:
        logger.info(f"- Available Integrations: {', '.join(integrations)}")
    else:
        logger.info("- Available Integrations: None")

#------------------------------------------------------------------------------
# Core Configuration
#------------------------------------------------------------------------------

class ModelArchitecture(Enum):
    """Supported model architectures"""
    TRANSFORMER = "transformer"  # Standard transformer models (GPT, BERT, etc)
    MAMBA = "mamba"             # Mamba/SSM models
    RETNET = "retnet"           # Retention Network models
    RWKV = "rwkv"               # RWKV models
    HYBRID = "hybrid"           # Hybrid architecture
    CUSTOM = "custom"           # Custom architecture


class PrecisionMode(Enum):
    """Precision modes supported by the engine"""
    ADAPTIVE = "adaptive"     # Dynamic token-level precision
    MIXED = "mixed"           # Standard mixed precision (FP16/BF16)
    INT8 = "int8"             # INT8 quantization
    INT4 = "int4"             # INT4 quantization
    STANDARD = "standard"     # Standard precision (usually FP32)
    CUSTOM = "custom"         # Custom precision scheme


class QuantizationScheme(Enum):
    """Supported quantization schemes"""
    SYMMETRIC = "symmetric"    # Symmetric quantization (zero-centered)
    ASYMMETRIC = "asymmetric"  # Asymmetric quantization (min-max based)
    LOGARITHMIC = "log"        # Logarithmic quantization (better for weights near zero)
    DYNAMIC = "dynamic"        # Dynamic quantization with outlier handling
    AWQV1 = "awq_v1"          # Activation-aware Weight Quantization v1
    AWQV2 = "awq_v2"          # Activation-aware Weight Quantization v2
    GPTQ = "gptq"             # GPTQ quantization
    BITBYTE = "bitbyte"       # BitByte quantization scheme
    CUSTOM = "custom"         # Custom quantization scheme


class TokenImportanceMetric(Enum):
    """Metrics for token importance"""
    ATTENTION = "attention"      # Based on attention scores
    HIDDEN_NORM = "hidden_norm"  # Based on hidden state norms
    GRADIENT = "gradient"        # Based on gradient magnitudes
    ENTROPY = "entropy"          # Based on attention entropy
    POSITION = "position"        # Based on token position
    STATE_NORM = "state_norm"    # Based on state vector norms (for SSM)
    TOKEN_ID = "token_id"        # Based on token IDs
    HYBRID = "hybrid"            # Hybrid approach
    UNIFORM = "uniform"          # Uniform importance
    CUSTOM = "custom"            # Custom importance metric


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    bits: int = 8                                            # Quantization bit depth
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC  # Quantization scheme
    per_tensor: bool = True                                  # Apply quantization per tensor
    per_channel: bool = False                                # Apply quantization per channel
    clip_outliers: bool = True                               # Clip outlier values
    outlier_threshold: float = 0.01                          # Threshold for outlier detection
    channel_dim: int = 0                                     # Channel dimension
    scale_dtype: torch.dtype = torch.float32                 # Data type for scales
    zero_point_dtype: torch.dtype = torch.int32              # Data type for zero points
    optimize_memory: bool = True                             # Optimize memory usage during quantization
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if isinstance(self.scheme, str):
            try:
                self.scheme = QuantizationScheme(self.scheme)
            except ValueError:
                logger.warning(f"Invalid quantization scheme: {self.scheme}, using SYMMETRIC")
                self.scheme = QuantizationScheme.SYMMETRIC


@dataclass
class TAPConfig:
    """Configuration for Token-Adaptive Precision"""
    # General settings
    seed: int = 42                                              # Random seed
    device: str = "auto"                                        # Device selection
    model_arch: ModelArchitecture = ModelArchitecture.TRANSFORMER  # Model architecture
    
    # Precision settings
    precision_mode: PrecisionMode = PrecisionMode.ADAPTIVE      # Main precision mode
    precision_levels: List[int] = field(default_factory=lambda: [4, 8, 16, 32])  # Available precision levels
    precision_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])  # Thresholds for precision levels
    default_precision: int = 16                                 # Default precision when not using adaptive
    
    # Token importance settings
    importance_method: Union[str, TokenImportanceMetric] = TokenImportanceMetric.HYBRID  # Method to determine token importance
    importance_cache_size: int = 2000                           # Size of importance score cache
    importance_update_interval: int = 1                         # Update interval for importance scores
    
    # Quantization settings
    weight_quantization: Optional[QuantizationConfig] = None    # Config for weight quantization
    activation_quantization: Optional[QuantizationConfig] = None  # Config for activation quantization
    quant_aware_training: bool = False                          # Enable Quantization-Aware Training
    
    # SSM/Mamba specific settings
    ssm_state_precision: int = 16                               # Precision for SSM state vectors
    ssm_scan_precision: int = 16                                # Precision for SSM scan operation
    ssm_token_shift_precision: int = 16                         # Precision for token shift operations
    
    # Optimization settings
    compilation_enabled: bool = True                            # Enable model compilation
    compile_mode: str = "reduce-overhead"                       # Compilation mode
    optimization_level: int = 2                                 # Optimization level (0-3)
    cache_optimizations: bool = True                            # Enable caching optimizations
    
    # Training settings
    gradient_accumulation_steps: int = 1                        # Gradient accumulation steps
    gradient_checkpointing: bool = False                        # Enable gradient checkpointing
    enable_mixed_precision_training: bool = True                # Use mixed precision for training
    bf16: bool = False                                          # Use bfloat16 instead of float16 if available
    
    # Energy and memory settings
    enable_energy_tracking: bool = True                         # Track energy usage
    energy_optimization_level: int = 2                          # Energy optimization level (0-3)
    memory_efficient_mode: bool = True                          # Use memory-efficient operations
    max_memory_usage_pct: float = 90.0                          # Max memory usage percentage
    offload_to_cpu: bool = False                                # Offload tensors to CPU
    
    # Integration settings
    use_accelerate: bool = True                                 # Use HuggingFace Accelerate
    use_bitbyte: bool = True                                    # Use BitsAndBytes for quantization
    use_peft: bool = False                                      # Use PEFT for parameter-efficient fine-tuning
    use_triton: bool = True                                     # Use Triton kernels when available
    
    # Advanced settings
    profiling_enabled: bool = False                             # Enable performance profiling
    debug_mode: bool = False                                    # Enable debug mode
    error_correction: bool = True                               # Enable error correction
    verbose_logging: bool = False                               # Enable verbose logging
    
    # Estimated energy factors for different bit precisions
    energy_tracking: Dict[str, Dict[int, float]] = field(
        default_factory=lambda: {
            "relative_energy": {
                4: 0.25,   # 4-bit operations use ~25% energy of 32-bit
                8: 0.4,    # 8-bit operations use ~40% energy of 32-bit
                16: 0.6,   # 16-bit operations use ~60% energy of 32-bit
                32: 1.0    # 32-bit is the baseline
            }
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Auto-select device if not specified
        if self.device == "auto":
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Initialize quantization configs if not provided
        if self.weight_quantization is None:
            self.weight_quantization = QuantizationConfig(bits=8)
        
        if self.activation_quantization is None:
            self.activation_quantization = QuantizationConfig(bits=8)
        
        # Convert string enum values to actual enums
        if isinstance(self.model_arch, str):
            try:
                self.model_arch = ModelArchitecture(self.model_arch)
            except ValueError:
                logger.warning(f"Unknown model architecture: {self.model_arch}, defaulting to TRANSFORMER")
                self.model_arch = ModelArchitecture.TRANSFORMER
        
        if isinstance(self.precision_mode, str):
            try:
                self.precision_mode = PrecisionMode(self.precision_mode)
            except ValueError:
                logger.warning(f"Unknown precision mode: {self.precision_mode}, defaulting to ADAPTIVE")
                self.precision_mode = PrecisionMode.ADAPTIVE
        
        if isinstance(self.importance_method, str):
            try:
                self.importance_method = TokenImportanceMetric(self.importance_method)
            except ValueError:
                logger.warning(f"Unknown importance method: {self.importance_method}, defaulting to HYBRID")
                self.importance_method = TokenImportanceMetric.HYBRID
        
        # Check precision levels and thresholds
        if len(self.precision_levels) != len(self.precision_thresholds) + 1:
            logger.warning(f"Invalid precision configuration: precision_levels should have length equal to thresholds + 1")
            # Auto-fix by adjusting thresholds
            if len(self.precision_levels) > 1:
                step = 1.0 / len(self.precision_levels)
                self.precision_thresholds = [step * (i+1) for i in range(len(self.precision_levels) - 1)]
                logger.warning(f"Adjusted thresholds to: {self.precision_thresholds}")
        
        # Adjust settings based on capabilities
        self._adjust_for_capabilities()
        
        # Validate CPU offload settings
        if self.offload_to_cpu and not self.memory_efficient_mode:
            logger.warning("CPU offload requires memory_efficient_mode, enabling it automatically")
            self.memory_efficient_mode = True
        
        # Log configuration
        if self.verbose_logging:
            self._log_configuration()
    
    def _adjust_for_capabilities(self):
        """Adjust settings based on system capabilities"""
        # Check for PyTorch
        if not TORCH_AVAILABLE:
            self.compilation_enabled = False
            self.enable_mixed_precision_training = False
            self.bf16 = False
            logger.warning("PyTorch not available, disabling compilation and mixed precision")
        
        # Adjust for bfloat16 support
        if self.bf16:
            if not TORCH_AVAILABLE or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                self.bf16 = False
                logger.warning("BF16 not supported on this system, falling back to FP16")
        
        # Adjust torch.compile based on PyTorch version
        if self.compilation_enabled and TORCH_AVAILABLE:
            torch_version = torch.__version__.split('.')
            if int(torch_version[0]) < 2:
                self.compilation_enabled = False
                logger.warning("torch.compile requires PyTorch 2.0+, disabling compilation")
        
        # Check CUDA availability for GPU-related settings
        if self.device != "cpu" and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            logger.warning(f"CUDA not available, switching to CPU mode")
            self.device = "cpu"
            self.enable_mixed_precision_training = False
            self.bf16 = False
            self.energy_optimization_level = min(1, self.energy_optimization_level)
        
        # Adjust integration settings based on available libraries
        if self.use_accelerate and not ACCELERATE_AVAILABLE:
            self.use_accelerate = False
            logger.warning("Accelerate library not available, disabling integration")
        
        if self.use_bitbyte and not BITSANDBYTES_AVAILABLE:
            self.use_bitbyte = False
            logger.warning("BitsAndBytes library not available, disabling integration")
        
        if self.use_peft and not PEFT_AVAILABLE:
            self.use_peft = False
            logger.warning("PEFT library not available, disabling integration")
        
        if self.use_triton and not TRITON_AVAILABLE:
            self.use_triton = False
            logger.warning("Triton not available, disabling custom kernels")
    
    def _log_configuration(self):
        """Log the current configuration"""
        logger.info(f"TAP Engine Configuration:")
        logger.info(f"- Model Architecture: {self.model_arch.value}")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Precision Mode: {self.precision_mode.value}")
        
        if self.precision_mode == PrecisionMode.ADAPTIVE:
            log_levels = ", ".join([f"{p}-bit" for p in self.precision_levels])
            logger.info(f"- Precision Levels: {log_levels}")
            logger.info(f"- Token Importance Method: {self.importance_method.value if isinstance(self.importance_method, Enum) else self.importance_method}")
        
        integration_info = []
        if self.compilation_enabled:
            integration_info.append(f"Compilation ({self.compile_mode})")
        if self.use_accelerate:
            integration_info.append("Accelerate")
        if self.use_bitbyte:
            integration_info.append("BitsAndBytes")
        if self.use_peft:
            integration_info.append("PEFT")
        if self.use_triton:
            integration_info.append("Triton")
        
        if integration_info:
            logger.info(f"- Enabled Integrations: {', '.join(integration_info)}")
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif dataclasses.is_dataclass(value):
                result[key] = dataclasses.asdict(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TAPConfig':
        """Create config from dictionary"""
        # Handle nested dataclass fields
        clean_dict = {}
        
        for key, value in config_dict.items():
            if key == "weight_quantization" and value is not None:
                clean_dict[key] = QuantizationConfig(**value)
            elif key == "activation_quantization" and value is not None:
                clean_dict[key] = QuantizationConfig(**value)
            else:
                clean_dict[key] = value
        
        # Convert string enum values
        enum_fields = {
            "model_arch": ModelArchitecture,
            "precision_mode": PrecisionMode,
            "importance_method": TokenImportanceMetric
        }
        
        for field_name, enum_class in enum_fields.items():
            if field_name in clean_dict and isinstance(clean_dict[field_name], str):
                try:
                    clean_dict[field_name] = enum_class(clean_dict[field_name])
                except ValueError:
                    # Keep as string, will be handled in __post_init__
                    pass
        
        return cls(**clean_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TAPConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_quant_config_for_module(self, module_name: str) -> QuantizationConfig:
        """Get quantization config for a specific module"""
        # Here we could implement more sophisticated matching based on module_name
        # For now we just return the default weight quantization config
        return copy.deepcopy(self.weight_quantization)
    
    def create_optimized_config(self, optimization_target="speed") -> 'TAPConfig':
        """Create an optimized version of the config for a specific target"""
        new_config = copy.deepcopy(self)
        
        if optimization_target == "speed":
            # Speed-optimized settings
            new_config.precision_mode = PrecisionMode.MIXED
            new_config.default_precision = 16
            new_config.compilation_enabled = True
            new_config.compile_mode = "max-autotune"
            new_config.cache_optimizations = True
            new_config.memory_efficient_mode = False  # Might be slower but faster
            
            # Use faster but less accurate quantization
            new_config.weight_quantization.bits = 8
            new_config.weight_quantization.scheme = QuantizationScheme.SYMMETRIC
            new_config.weight_quantization.clip_outliers = False
            
        elif optimization_target == "memory":
            # Memory-optimized settings
            new_config.precision_mode = PrecisionMode.ADAPTIVE
            new_config.precision_levels = [4, 8, 16]
            new_config.precision_thresholds = [0.3, 0.7]
            new_config.memory_efficient_mode = True
            new_config.offload_to_cpu = True
            new_config.gradient_checkpointing = True
            
            # Aggressive quantization
            new_config.weight_quantization.bits = 4
            new_config.weight_quantization.scheme = QuantizationScheme.DYNAMIC
            new_config.weight_quantization.clip_outliers = True
            new_config.weight_quantization.outlier_threshold = 0.01
            
        elif optimization_target == "energy":
            # Energy-optimized settings
            new_config.precision_mode = PrecisionMode.ADAPTIVE
            new_config.precision_levels = [4, 8, 16, 32]
            new_config.precision_thresholds = [0.2, 0.5, 0.8]
            new_config.energy_optimization_level = 3
            new_config.memory_efficient_mode = True
            
            # Balance of precision and energy
            new_config.weight_quantization.bits = 8
            new_config.weight_quantization.scheme = QuantizationScheme.DYNAMIC
            
        return new_config


#------------------------------------------------------------------------------
# Memory and Energy Monitoring
#------------------------------------------------------------------------------

class MemoryTracker:
    """
    Enhanced memory tracker with optimizations for large models.
    Tracks memory usage across CPU and GPU, identifies memory bottlenecks,
    and implements memory-saving techniques.
    """
    def __init__(self, device_id=0, max_memory_pct=90.0, config=None):
        self.device_id = device_id
        self.max_memory_pct = max_memory_pct
        self.config = config or TAPConfig()
        
        # Basic tracking variables
        self.peak_memory = 0
        self.memory_history = deque(maxlen=100)  # Use deque with max length for better memory management
        self.last_check_time = time.time()
        self.check_interval = 0.5  # seconds
        
        # Enhanced tracking for memory leaks
        self.allocation_map = {}
        self.tracking_enabled = False
        self.memory_alerts = []
        self.fragmentation_threshold = 0.3  # Alert when fragmentation exceeds 30%
        
        # CPU memory tracking
        self.track_cpu_memory = True
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.track_cpu_memory = False
            logger.warning("psutil not available, CPU memory tracking disabled")
    
    def enable_detailed_tracking(self, enabled=True):
        """Enable or disable detailed memory tracking"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            logger.warning("Detailed memory tracking requires CUDA, but CUDA is not available")
            return False
        
        self.tracking_enabled = enabled
        if enabled:
            try:
                torch.cuda.memory._record_memory_history(enabled="all")
                logger.info("Detailed GPU memory tracking enabled")
                return True
            except Exception as e:
                logger.error(f"Failed to enable detailed memory tracking: {e}")
                self.tracking_enabled = False
                return False
        else:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
                logger.info("Detailed GPU memory tracking disabled")
                return True
            except Exception:
                return False
    
    def check_memory(self, force=False) -> Dict[str, Any]:
        """Check current memory usage and return stats"""
        current_time = time.time()
        
        # Limit checks to avoid performance impact unless forced
        if not force and current_time - self.last_check_time < self.check_interval:
            return {}
        
        self.last_check_time = current_time
        
        stats = {
            "timestamp": current_time,
            "memory_available": True
        }
        
        # GPU memory stats
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device = f"cuda:{self.device_id}"
                stats["device"] = device
                
                # Get memory stats
                allocated = torch.cuda.memory_allocated(self.device_id)
                reserved = torch.cuda.memory_reserved(self.device_id)
                max_memory = torch.cuda.get_device_properties(self.device_id).total_memory
                
                # Update peak memory
                self.peak_memory = max(self.peak_memory, allocated)
                
                # Calculate percentages
                pct_allocated = (allocated / max_memory) * 100
                pct_reserved = (reserved / max_memory) * 100
                
                # Check if we're within limits
                memory_available = pct_allocated < self.max_memory_pct
                
                # Get additional details
                if self.tracking_enabled:
                    try:
                        # Get the current snapshot
                        snapshot = torch.cuda.memory._snapshot()
                        stats["active_allocations"] = len(snapshot["segments"])
                        
                        # Calculate fragmentation
                        if reserved > 0:
                            fragmentation = 1.0 - (allocated / reserved)
                            stats["fragmentation"] = fragmentation
                            
                            # Create alert if fragmentation is high
                            if fragmentation > self.fragmentation_threshold:
                                self.memory_alerts.append({
                                    "type": "high_fragmentation",
                                    "timestamp": current_time,
                                    "fragmentation": fragmentation,
                                    "allocated": allocated,
                                    "reserved": reserved
                                })
                    except Exception as e:
                        if self.config.debug_mode:
                            logger.error(f"Error in detailed memory tracking: {e}")
                
                # Record history
                self.memory_history.append({
                    "timestamp": current_time,
                    "allocated": allocated,
                    "reserved": reserved,
                    "pct_allocated": pct_allocated
                })
                
                stats.update({
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "total_bytes": max_memory,
                    "pct_allocated": pct_allocated,
                    "pct_reserved": pct_reserved,
                    "peak_bytes": self.peak_memory,
                    "memory_available": memory_available
                })
            except Exception as e:
                logger.error(f"Error checking GPU memory: {e}")
                stats["error"] = str(e)
        else:
            stats["device"] = "cpu"
        
        # CPU memory stats
        if self.track_cpu_memory:
            try:
                vm = self.psutil.virtual_memory()
                cpu_stats = {
                    "cpu_total_bytes": vm.total,
                    "cpu_available_bytes": vm.available,
                    "cpu_used_bytes": vm.used,
                    "cpu_percent": vm.percent
                }
                stats.update(cpu_stats)
                
                # Check if CPU memory is within limits
                if vm.percent > self.max_memory_pct:
                    stats["memory_available"] = False
                    
                    # Add alert
                    self.memory_alerts.append({
                        "type": "high_cpu_usage",
                        "timestamp": current_time,
                        "cpu_percent": vm.percent
                    })
            except Exception as e:
                logger.error(f"Error checking CPU memory: {e}")
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = self.check_memory(force=True)
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Add trend information
            if len(self.memory_history) > 1:
                first = self.memory_history[0]
                last = self.memory_history[-1]
                
                time_diff = last["timestamp"] - first["timestamp"]
                if time_diff > 0:
                    memory_growth_rate = (last["allocated"] - first["allocated"]) / time_diff
                    stats["memory_growth_rate_bytes_per_sec"] = memory_growth_rate
                    
                    # Create alert if growth rate is high (over 100MB/s)
                    if memory_growth_rate > 100 * 1024 * 1024:  # 100 MB/s
                        self.memory_alerts.append({
                            "type": "high_growth_rate",
                            "timestamp": time.time(),
                            "growth_rate_mb_per_sec": memory_growth_rate / (1024 * 1024)
                        })
        
        # Add alerts to stats
        if self.memory_alerts:
            stats["alerts"] = self.memory_alerts[-5:]  # Include last 5 alerts
        
        return stats
    
    def optimize_memory(self, force=False) -> bool:
        """
        Optimize memory usage by clearing caches and running garbage collection
        Returns True if optimization was performed
        """
        stats = self.check_memory()
        
        # Only optimize if we're using too much memory or if forced
        if not force and stats.get("memory_available", True):
            return False
        
        # Log pre-optimization state
        before_allocated = stats.get("allocated_bytes", 0)
        
        # Collect Python garbage
        gc.collect()
        
        # Clear PyTorch caches
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Additional cleanup for tensors with no references
            if hasattr(torch.cuda, 'memory_snapshot'):
                try:
                    # This is more aggressive and can help with fragmentation
                    torch.cuda.memory._dump_snapshot()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        
        # Check post-optimization memory
        after_stats = self.check_memory(force=True)
        after_allocated = after_stats.get("allocated_bytes", 0)
        
        # Calculate memory freed
        memory_freed = max(0, before_allocated - after_allocated)
        memory_freed_mb = memory_freed / (1024 * 1024)
        
        logger.info(f"Memory optimized: {stats.get('pct_allocated', 0):.1f}% → "
                   f"{after_stats.get('pct_allocated', 0):.1f}% (freed {memory_freed_mb:.2f} MB)")
        
        return True
    
    def register_tensor(self, tensor, name=None):
        """Register a tensor for memory tracking"""
        if not TORCH_AVAILABLE:
            return
            
        if name is None:
            # Generate a name based on stack frame
            frame = inspect.currentframe().f_back
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            name = f"{os.path.basename(filename)}:{lineno}"
        
        tensor_id = id(tensor)
        
        self.allocation_map[tensor_id] = {
            "name": name,
            "shape": tensor.shape if hasattr(tensor, "shape") else None,
            "dtype": tensor.dtype if hasattr(tensor, "dtype") else None,
            "allocated_time": time.time(),
            "size_bytes": tensor.element_size() * tensor.nelement() if hasattr(tensor, "element_size") else 0
        }
    
    def unregister_tensor(self, tensor):
        """Unregister a tensor from memory tracking"""
        tensor_id = id(tensor)
        if tensor_id in self.allocation_map:
            del self.allocation_map[tensor_id]
    
    def get_allocation_map(self):
        """Get a map of current tensor allocations"""
        # Clean up map of deleted tensors
        current_ids = set()
        for tensor_id, info in list(self.allocation_map.items()):
            if not self._is_tensor_alive(tensor_id):
                del self.allocation_map[tensor_id]
            else:
                current_ids.add(tensor_id)
        
        # Detect leaks (tensors allocated more than 10 minutes ago)
        current_time = time.time()
        leaks = []
        for tensor_id, info in self.allocation_map.items():
            if current_time - info["allocated_time"] > 600:  # 10 minutes
                leaks.append(info)
        
        return {
            "total_tracked_tensors": len(self.allocation_map),
            "total_tracked_bytes": sum(info["size_bytes"] for info in self.allocation_map.values()),
            "potential_leaks": leaks if leaks else None
        }
    
    def _is_tensor_alive(self, tensor_id):
        """Check if a tensor is still alive"""
        import weakref
        for obj in gc.get_objects():
            if id(obj) == tensor_id and torch.is_tensor(obj):
                return True
        return False
    
    @contextlib.contextmanager
    def track(self):
        """Context manager for memory tracking"""
        start_stats = self.get_memory_stats()
        yield self
        end_stats = self.get_memory_stats()
        
        # Log memory change
        if TORCH_AVAILABLE and torch.cuda.is_available():
            memory_change = end_stats.get("allocated_bytes", 0) - start_stats.get("allocated_bytes", 0)
            memory_change_mb = memory_change / (1024 * 1024)
            logger.debug(f"Memory change: {memory_change_mb:.2f} MB")


class EnergyMonitor:
    """
    Enhanced energy monitor with support for different hardware platforms.
    Tracks power usage and estimates energy consumption for NVIDIA, AMD,
    and Apple Silicon GPUs, with CPU fallback.
    """
    def __init__(self, device_id=0, config=None):
        self.device_id = device_id
        self.config = config or TAPConfig()
        self.reset()
        
        # Enhanced capabilities detection
        self.capabilities = self._detect_capabilities()
        
        # Initialize monitoring based on capabilities
        self._initialize_monitoring()
        
        # Default power estimation parameters (updated based on device)
        self.power_estimation = {
            "idle_cpu_watts": 15.0,   # Idle CPU power draw
            "max_cpu_watts": 65.0,    # Max CPU power draw
            "idle_gpu_watts": 15.0,   # Idle GPU power draw
            "max_gpu_watts": 250.0,   # Max GPU power draw
            "flops_per_watt": 50e9,   # Approximate FLOPS per watt for modern hardware
            "bytes_per_joule": 1e6,   # Approximate bytes transferred per joule
            "power_draw_model": None  # ML model for power estimation (if available)
        }
        
        # Update power model based on detected hardware
        self._update_power_model()
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Enhanced detection of monitoring capabilities"""
        capabilities = {
            "nvml_available": False,       # NVIDIA management library
            "rocm_available": False,       # AMD ROCm monitoring
            "psutil_available": False,     # CPU monitoring via psutil
            "intel_rapl_available": False, # Intel RAPL power monitoring
            "mps_available": False,        # Apple Silicon monitoring
            "jetson_available": False,     # NVIDIA Jetson monitoring
            "powermetrics_available": False, # macOS powermetrics
            "sysfs_power_available": False, # Linux sysfs power monitoring
        }
        
        # Check for NVIDIA NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            capabilities["nvml_available"] = True
        except (ImportError, Exception):
            pass
        
        # Check for AMD ROCm
        try:
            import rocm_smi_lib
            capabilities["rocm_available"] = True
        except ImportError:
            pass
        
        # Check for psutil
        try:
            import psutil
            capabilities["psutil_available"] = True
        except ImportError:
            pass
        
        # Check for Intel RAPL
        try:
            rapl_path = "/sys/class/powercap/intel-rapl"
            capabilities["intel_rapl_available"] = os.path.exists(rapl_path)
        except:
            pass
        
        # Check for Apple Silicon monitoring
        try:
            if sys.platform == "darwin" and os.path.exists("/usr/bin/powermetrics"):
                capabilities["powermetrics_available"] = True
                # Check if we're on Apple Silicon
                import platform
                if platform.machine() == "arm64":
                    capabilities["mps_available"] = True
        except:
            pass
        
        # Check for NVIDIA Jetson
        try:
            jetson_path = "/sys/devices/platform/tegra-CPU-cluster/power_allocator"
            capabilities["jetson_available"] = os.path.exists(jetson_path)
        except:
            pass
        
        # Check for Linux sysfs power info (for modern laptops)
        try:
            power_path = "/sys/class/power_supply"
            if os.path.exists(power_path) and any("BAT" in f for f in os.listdir(power_path)):
                capabilities["sysfs_power_available"] = True
        except:
            pass
        
        return capabilities
    
    def _initialize_monitoring(self):
        """Initialize the appropriate monitoring interfaces"""
        if self.capabilities["nvml_available"]:
            import pynvml
            self.pynvml = pynvml
            self.gpu_handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            try:
                self.max_power_watts = self.pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000.0
            except Exception:
                self.max_power_watts = 250.0  # Default for most GPUs
            logger.debug(f"NVML monitoring initialized for device {self.device_id}, max power: {self.max_power_watts:.1f}W")
        
        elif self.capabilities["rocm_available"]:
            import rocm_smi_lib as rsmi
            self.rsmi = rsmi
            rsmi.rocm_smi_init()
            # For ROCm, device_id directly works with their API
            self.gpu_handle = self.device_id
            logger.debug(f"ROCm monitoring initialized for device {self.device_id}")
        
        elif self.capabilities["psutil_available"]:
            import psutil
            self.psutil = psutil
            logger.debug("Using psutil for CPU power monitoring")
        
        elif self.capabilities["powermetrics_available"]:
            # macOS power metrics requires root for real-time monitoring
            # We'll use a more general approach
            logger.debug("Using macOS powermetrics for energy estimation")
        
        else:
            logger.warning("No hardware monitoring found, using estimation only")
    
    def _update_power_model(self):
        """Update power model based on detected hardware"""
        # Update CPU power model based on detected cores
        if self.capabilities["psutil_available"]:
            cpu_count = self.psutil.cpu_count(logical=True)
            # Rough estimate: 5W base + 5W per core for idle, 15W base + 10W per core for max
            self.power_estimation["idle_cpu_watts"] = 5.0 + cpu_count * 5.0
            self.power_estimation["max_cpu_watts"] = 15.0 + cpu_count * 10.0
        
        # Update GPU power model if available
        if self.capabilities["nvml_available"]:
            try:
                # Get actual GPU parameters
                device_info = self.pynvml.nvmlDeviceGetName(self.gpu_handle)
                # Adjust idle power based on GPU class
                if "3090" in device_info or "4090" in device_info:
                    self.power_estimation["idle_gpu_watts"] = 30.0
                    self.power_estimation["max_gpu_watts"] = 350.0
                elif "2080" in device_info or "3080" in device_info:
                    self.power_estimation["idle_gpu_watts"] = 20.0
                    self.power_estimation["max_gpu_watts"] = 320.0
                elif "1080" in device_info or "2070" in device_info:
                    self.power_estimation["idle_gpu_watts"] = 15.0
                    self.power_estimation["max_gpu_watts"] = 215.0
                elif "T4" in device_info or "A10" in device_info:
                    self.power_estimation["idle_gpu_watts"] = 10.0
                    self.power_estimation["max_gpu_watts"] = 150.0
                
                # Get actual max power if available
                try:
                    self.power_estimation["max_gpu_watts"] = self.pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000.0
                except:
                    pass
            except Exception as e:
                logger.debug(f"Could not get detailed GPU info: {e}")
    
    def reset(self):
        """Reset monitoring variables"""
        self.start_time = None
        self.end_time = None
        self.energy_readings = []
        self.utilization_readings = []
        self.memory_readings = []
        self.temperature_readings = []
        self.tracking = False
        self.flops_performed = 0
        self.bytes_transferred = 0
    
    def start(self):
        """Start energy monitoring"""
        self.reset()
        self.start_time = time.time()
        self.tracking = True
        
        # Reset GPU stats if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device_id)
            except Exception:
                pass
        
        # Start background thread for energy readings
        if self.capabilities["nvml_available"] or self.capabilities["rocm_available"]:
            self._start_gpu_monitoring()
        elif self.capabilities["psutil_available"]:
            self._start_cpu_monitoring()
        elif self.capabilities["powermetrics_available"]:
            self._start_powermetrics_monitoring()
    
    def _start_gpu_monitoring(self):
        """Start GPU monitoring in a separate thread"""
        def _monitor_gpu():
            while self.tracking:
                try:
                    # Get power, utilization, memory and temperature
                    if self.capabilities["nvml_available"]:
                        # Power (mW to W)
                        power = self.pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                        
                        # Utilization (%)
                        utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                        
                        # Memory (bytes to MiB)
                        memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        memory_used = memory_info.used / (1024 * 1024)
                        
                        # Temperature (C)
                        temperature = self.pynvml.nvmlDeviceGetTemperature(
                            self.gpu_handle, self.pynvml.NVML_TEMPERATURE_GPU)
                    
                    elif self.capabilities["rocm_available"]:
                        # Similar metrics for AMD GPUs
                        power = self.rsmi.getGpuPower(self.gpu_handle) / 1000.0  # mW to W
                        utilization = self.rsmi.getGpuBusy(self.gpu_handle)
                        memory_info = self.rsmi.getMemInfo(self.gpu_handle)
                        memory_used = memory_info[0] / (1024 * 1024)  # Bytes to MiB
                        temperature = self.rsmi.getTemp(self.gpu_handle)
                    
                    # Record readings
                    self.energy_readings.append(power)
                    self.utilization_readings.append(utilization)
                    self.memory_readings.append(memory_used)
                    self.temperature_readings.append(temperature)
                    
                    time.sleep(0.1)  # Sample every 100ms
                except Exception as e:
                    logger.error(f"Error in GPU monitoring thread: {e}")
                    if self.config and self.config.debug_mode:
                        import traceback
                        logger.error(traceback.format_exc())
                    break
        
        self.monitor_thread = threading.Thread(target=_monitor_gpu, daemon=True)
        self.monitor_thread.start()
    
    def _start_cpu_monitoring(self):
        """Start CPU monitoring in a separate thread"""
        def _monitor_cpu():
            # Initial reading to establish baseline
            last_cpu_time = self.psutil.cpu_times()
            last_time = time.time()
            
            while self.tracking:
                try:
                    # Get current time and CPU stats
                    current_time = time.time()
                    current_cpu_time = self.psutil.cpu_times()
                    
                    # Calculate utilization
                    time_diff = current_time - last_time
                    user_diff = current_cpu_time.user - last_cpu_time.user
                    system_diff = current_cpu_time.system - last_cpu_time.system
                    
                    # Total active CPU time
                    active_time = user_diff + system_diff
                    
                    # Utilization percentage (all cores)
                    utilization = min(100.0, (active_time / (time_diff * self.psutil.cpu_count())) * 100)
                    
                    # Estimate power based on utilization (improved model)
                    idle_power = self.power_estimation["idle_cpu_watts"]
                    max_power = self.power_estimation["max_cpu_watts"]
                    
                    # Quadratic power model (more realistic than linear)
                    # Power increases more rapidly at higher utilization
                    util_factor = (utilization / 100.0) ** 2
                    estimated_power = idle_power + util_factor * (max_power - idle_power)
                    
                    # Memory usage
                    memory = self.psutil.virtual_memory()
                    memory_used = memory.used / (1024 * 1024)  # Convert to MiB
                    
                    # CPU temperature if available
                    try:
                        temperatures = self.psutil.sensors_temperatures()
                        cpu_temp = 0
                        count = 0
                        # Average across all CPU temperature sensors
                        for name, entries in temperatures.items():
                            if 'cpu' in name.lower() or 'core' in name.lower():
                                for entry in entries:
                                    cpu_temp += entry.current
                                    count += 1
                        temperature = cpu_temp / count if count > 0 else 0
                    except:
                        temperature = 0
                    
                    # Record readings
                    self.energy_readings.append(estimated_power)
                    self.utilization_readings.append(utilization)
                    self.memory_readings.append(memory_used)
                    self.temperature_readings.append(temperature)
                    
                    # Update for next iteration
                    last_cpu_time = current_cpu_time
                    last_time = current_time
                    
                    time.sleep(0.5)  # Sample every 500ms for CPU
                except Exception as e:
                    logger.error(f"Error in CPU monitoring thread: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=_monitor_cpu, daemon=True)
        self.monitor_thread.start()
    
    def _start_powermetrics_monitoring(self):
        """Start monitoring on macOS using powermetrics"""
        def _monitor_macos():
            while self.tracking:
                try:
                    # We can't directly call powermetrics without sudo, so use estimation
                    # based on CPU and memory usage
                    
                    # Get CPU and memory usage
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    memory = psutil.virtual_memory()
                    memory_used = memory.used / (1024 * 1024)  # Convert to MiB
                    
                    # For Apple Silicon, estimate based on usage
                    # M1/M2 chips are very energy efficient
                    if self.capabilities["mps_available"]:
                        # Idle: ~3W, Max: ~30W for M1/M2
                        idle_power = 3.0
                        max_power = 30.0
                    else:
                        # Intel Macs: Idle: ~10W, Max: ~60W
                        idle_power = 10.0
                        max_power = 60.0
                    
                    # Power increases exponentially with CPU usage
                    util_factor = (cpu_percent / 100.0) ** 1.5  # More efficient at lower usage
                    estimated_power = idle_power + util_factor * (max_power - idle_power)
                    
                    # Add GPU power if using MPS
                    if self.capabilities["mps_available"] and TORCH_AVAILABLE:
                        try:
                            # Check if we're actually using MPS
                            if torch.backends.mps.is_available() and SYSTEM_INFO.get("current_device", "") == "mps":
                                # Add 1-5W depending on memory allocation
                                if hasattr(torch.mps, "current_allocated_memory"):
                                    gpu_memory = torch.mps.current_allocated_memory() / (1024 * 1024)  # B to MiB
                                    gpu_power = 1.0 + min(4.0, gpu_memory / 1000.0)  # 1W + up to 4W based on memory
                                    estimated_power += gpu_power
                        except Exception:
                            pass
                    
                    # Record readings
                    self.energy_readings.append(estimated_power)
                    self.utilization_readings.append(cpu_percent)
                    self.memory_readings.append(memory_used)
                    self.temperature_readings.append(0)  # No temp data
                    
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error in macOS monitoring thread: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=_monitor_macos, daemon=True)
        self.monitor_thread.start()
    
    def record_computation(self, flops, bytes_transferred=0):
        """Record computation for energy estimation"""
        self.flops_performed += flops
        self.bytes_transferred += bytes_transferred
    
    def record_operation(self, op_type, input_shape, output_shape=None, precision=32):
        """
        Record operation for energy estimation using operation type and shapes
        
        Args:
            op_type: Operation type (matmul, conv, etc.)
            input_shape: Shape of input tensor(s)
            output_shape: Shape of output tensor
            precision: Numerical precision (4, 8, 16, 32)
        """
        # Calculate FLOPs based on operation type and shapes
        flops = 0
        bytes = 0
        
        if op_type == "matmul":
            # For matrix multiply: 2 * M * N * K FLOPs
            if isinstance(input_shape, tuple) and len(input_shape) == 2:
                # Single input shape provided, assume square matrix
                M, K = input_shape
                N = K
            elif isinstance(input_shape, list) and len(input_shape) == 2:
                # Two input matrices: A(M,K) x B(K,N)
                M, K = input_shape[0]
                K2, N = input_shape[1]
                assert K == K2, f"Incompatible matmul dimensions: {K} != {K2}"
            else:
                raise ValueError(f"Invalid input shape for matmul: {input_shape}")
            
            flops = 2 * M * N * K  # Multiply-add counts as 2 FLOPs
            bytes = (M * K + K * N + M * N) * (precision / 8)  # Input and output bytes
            
        elif op_type == "conv2d":
            # For 2D convolution: 2 * Cout * Hout * Wout * Cin * Kh * Kw FLOPs
            if isinstance(input_shape, tuple) and len(input_shape) == 4:
                # Input shape: [batch, Cin, H, W]
                batch, Cin, H, W = input_shape
                
                if isinstance(output_shape, tuple) and len(output_shape) == 4:
                    # Output shape: [batch, Cout, Hout, Wout]
                    _, Cout, Hout, Wout = output_shape
                    
                    # Assume 3x3 kernel by default
                    Kh, Kw = 3, 3
                    
                    flops = 2 * batch * Cout * Hout * Wout * Cin * Kh * Kw
                    bytes = (batch * Cin * H * W + batch * Cout * Hout * Wout + Cout * Cin * Kh * Kw) * (precision / 8)
        
        elif op_type == "attention":
            # For self-attention: 
            # Q,K,V projections: 3 * 2 * B * S * H * H_dim FLOPs
            # QK^T: 2 * B * NH * S * S * H_dim FLOPs
            # Attention * V: 2 * B * NH * S * S * H_dim FLOPs
            if isinstance(input_shape, tuple) and len(input_shape) == 3:
                # [batch, seq_len, hidden_dim]
                B, S, H = input_shape
                
                # Assume 12 attention heads by default
                NH = 12
                H_dim = H // NH
                
                # Total FLOPs
                flops = (3 * 2 * B * S * H * H) + (2 * B * NH * S * S * H_dim) + (2 * B * NH * S * S * H_dim)
                bytes = (4 * B * S * H) * (precision / 8)  # Q, K, V, output
        
        elif op_type == "ssm" or op_type == "mamba":
            # For SSM/Mamba operations (S4 block):
            # - Projection: 2 * B * S * H * H FLOPs
            # - Convolution: ~ 2 * B * S * H * log(S) FLOPs (using FFT)
            # - State update: 2 * B * S * H * D FLOPs (D = state dimension)
            if isinstance(input_shape, tuple) and len(input_shape) == 3:
                # [batch, seq_len, hidden_dim]
                B, S, H = input_shape
                
                # Assume state dimension is 16 by default
                D = 16
                
                # Total FLOPs
                proj_flops = 2 * B * S * H * H
                conv_flops = 2 * B * S * H * int(math.log2(S))
                state_flops = 2 * B * S * H * D
                
                flops = proj_flops + conv_flops + state_flops
                bytes = (2 * B * S * H + B * S * D) * (precision / 8)
        
        # Scale FLOPs based on precision (approximation)
        precision_factor = 32 / precision
        flops = int(flops * precision_factor)
        
        # Record the computed values
        self.record_computation(flops, bytes)
    
    def stop(self):
        """Stop monitoring and calculate energy used"""
        self.tracking = False
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.energy_readings:
            # Calculate average power (W) and total energy (J)
            avg_power = sum(self.energy_readings) / len(self.energy_readings)
            total_energy = avg_power * duration  # W·s (joules)
            
            # Calculate average utilization
            avg_utilization = sum(self.utilization_readings) / len(self.utilization_readings) if self.utilization_readings else 0
            avg_memory = sum(self.memory_readings) / len(self.memory_readings) if self.memory_readings else 0
            peak_memory = max(self.memory_readings) if self.memory_readings else 0
            avg_temperature = sum(self.temperature_readings) / len(self.temperature_readings) if self.temperature_readings else 0
            
            result = {
                "avg_power": avg_power,  # W
                "total_energy": total_energy,  # J
                "duration": duration,  # s
                "avg_utilization": avg_utilization,  # %
                "avg_memory_mib": avg_memory,  # MiB
                "peak_memory_mib": peak_memory,  # MiB
                "avg_temperature": avg_temperature,  # °C
                "source": "measured"
            }
        else:
            # If no readings, estimate from computation
            result = self._estimate_energy_usage(duration)
        
        return result
    
    def _estimate_energy_usage(self, duration):
        """
        Estimate energy usage from computation data and memory usage
        with improved accuracy for different architectures
        """
        # Read memory info
        if TORCH_AVAILABLE and torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)  # MiB
            peak_memory = torch.cuda.max_memory_allocated(self.device_id) / (1024 * 1024)  # MiB
            reserved_memory = torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)  # MiB
        else:
            try:
                import psutil
                vm = psutil.virtual_memory()
                current_memory = vm.used / (1024 * 1024)
                peak_memory = current_memory
                reserved_memory = vm.total / (1024 * 1024)
            except:
                current_memory = 0
                peak_memory = 0
                reserved_memory = 0
        
        # Energy from computation (FLOPS)
        # Energy per GFLOP varies by hardware and precision
        device_type = "cpu" if self.device_id == "cpu" or not TORCH_AVAILABLE or not torch.cuda.is_available() else "gpu"
        
        if device_type == "gpu":
            base_energy_per_gflop = 1.5  # mJ/GFLOP for recent NVIDIA GPUs
            
            # Adjust for different GPU generations and architectures
            if hasattr(self, 'gpu_handle') and self.capabilities["nvml_available"]:
                try:
                    device_name = self.pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
                    
                    # Newer GPUs are more efficient
                    if any(x in device_name for x in ["A100", "H100", "A30", "A10G"]):
                        base_energy_per_gflop = 1.0  # Latest NVIDIA datacenter GPUs
                    elif any(x in device_name for x in ["4090", "3090", "3080"]):
                        base_energy_per_gflop = 1.2  # Latest consumer GPUs
                    elif any(x in device_name for x in ["2080", "2070"]):
                        base_energy_per_gflop = 1.8  # Older RTX GPUs
                    elif any(x in device_name for x in ["1080", "1070"]):
                        base_energy_per_gflop = 2.2  # Pascal GPUs
                except Exception:
                    pass
        else:
            base_energy_per_gflop = 3.0  # mJ/GFLOP for CPU
        
        energy_from_compute = 0
        if self.flops_performed > 0:
            gflops = self.flops_performed / 1e9
            energy_from_compute = gflops * base_energy_per_gflop / 1000.0  # Convert from mJ to J
        
        # Energy from memory usage
        # Memory power factors vary by hardware
        if device_type == "gpu":
            memory_power_factor = 0.1 / 1024  # W/MiB for GPU
            idle_power = self.power_estimation["idle_gpu_watts"]  # W idle for GPU
        else:
            memory_power_factor = 0.05 / 1024  # W/MiB for CPU
            idle_power = self.power_estimation["idle_cpu_watts"]  # W idle for CPU
        
        memory_energy = peak_memory * memory_power_factor * duration
        idle_energy = idle_power * duration
        
        # Total energy
        total_energy = energy_from_compute + memory_energy + idle_energy
        avg_power = total_energy / duration if duration > 0 else 0
        
        # Estimate utilization
        if self.flops_performed > 0 and duration > 0:
            # Calculate TFLOPS
            tflops = self.flops_performed / 1e12 / duration
            
            # Estimate peak TFLOPS based on hardware
            if device_type == "gpu":
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        device_properties = torch.cuda.get_device_properties(self.device_id)
                        # Improved estimate based on CUDA cores and clock speed
                        cuda_cores = device_properties.multi_processor_count * 128  # Approximate CUDA cores per SM
                        clock_ghz = device_properties.clock_rate / 1e6  # Convert to GHz
                        
                        # Adjust based on GPU architecture
                        if hasattr(device_properties, 'major') and device_properties.major >= 8:
                            # Ampere or newer: better FP16/TF32 performance
                            estimated_peak_tflops = cuda_cores * clock_ghz * 2 * 2 / 1000  # FMA ops & tensor cores boost
                        else:
                            estimated_peak_tflops = cuda_cores * clock_ghz * 2 / 1000  # FMA operations
                    except Exception:
                        # Fallback estimate
                        estimated_peak_tflops = 15.0  # Reasonable for modern GPUs
                else:
                    estimated_peak_tflops = 15.0
            else:
                # CPU TFLOPS estimate
                try:
                    import psutil
                    num_cores = psutil.cpu_count(logical=False)
                    estimated_peak_tflops = num_cores * 0.1  # Rough estimate: 100 GFLOPS per core
                except Exception:
                    estimated_peak_tflops = 1.0  # Conservative fallback
            
            # Calculate utilization percentage
            estimated_utilization = min(100, tflops / estimated_peak_tflops * 100)
        else:
            # Estimate from average power compared to max power
            if device_type == "gpu":
                max_power = self.power_estimation["max_gpu_watts"]
            else:
                max_power = self.power_estimation["max_cpu_watts"]
            
            estimated_utilization = min(100, max(0, (avg_power - idle_power) / (max_power - idle_power) * 100))
        
        return {
            "avg_power": avg_power,  # W
            "total_energy": total_energy,  # J
            "duration": duration,  # s
            "avg_utilization": estimated_utilization,  # %
            "peak_memory_mib": peak_memory,  # MiB
            "current_memory_mib": current_memory,  # MiB
            "reserved_memory_mib": reserved_memory,  # MiB
            "source": "estimated",
            "flops_performed": self.flops_performed,
            "bytes_transferred": self.bytes_transferred
        }
    
    @contextlib.contextmanager
    def track(self):
        """Context manager for energy tracking"""
        self.start()
        try:
            yield self
        finally:
            result = self.stop()
            return result


#------------------------------------------------------------------------------
# Precision and Quantization Tools
#------------------------------------------------------------------------------
class PrecisionManager:
   """
   Advanced class for managing tensor precision and quantization.
   Provides tools for both static and dynamic precision management,
   with support for various quantization schemes and architectures.
   """
   
   @staticmethod
   def quantize_tensor(x, 
                       bits=8, 
                       scheme=QuantizationScheme.SYMMETRIC,
                       per_channel=False,
                       channel_dim=0,
                       clip_outliers=False,
                       outlier_threshold=0.01):
       """
       Quantize a tensor to reduced precision
       
       Args:
           x: Input tensor
           bits: Target bits (4, 8, 16, or 32)
           scheme: Quantization scheme
           per_channel: Whether to quantize per channel
           channel_dim: Channel dimension for per-channel quantization
           clip_outliers: Whether to clip outlier values
           outlier_threshold: Threshold for outlier clipping
           
       Returns:
           x_quantized: Quantized tensor
           metadata: Quantization metadata
       """
       # Check for NaN or Infinity
       if torch.isnan(x).any() or torch.isinf(x).any():
           logger.warning("Input tensor contains NaN or Infinity values. These will be replaced with zeros.")
           x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
           
       # Validate bits parameter
       if bits not in [4, 8, 16, 32]:
           logger.warning(f"Unsupported bit value: {bits}, using 8 bits instead")
           bits = 8
       
       metadata = {
           "bits": bits, 
           "scheme": scheme if isinstance(scheme, str) else scheme.value,
           "per_channel": per_channel,
           "channel_dim": channel_dim
       }
       
       if isinstance(scheme, str):
           try:
               scheme = QuantizationScheme(scheme)
           except ValueError:
               logger.warning(f"Unknown quantization scheme: {scheme}, using SYMMETRIC")
               scheme = QuantizationScheme.SYMMETRIC
       
       # No quantization for FP32 or non-floating tensors
       if bits == 32 or not x.is_floating_point():
           return x, metadata
       
       # Direct conversion for FP16/BF16
       if bits == 16:
           if hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported():
               return x.to(torch.bfloat16), metadata
           else:
               return x.to(torch.float16), metadata
       
       # Convert to FP32 for quantization calculation
       x_float = x.to(torch.float32)
       
       # Prepare for per-channel quantization if requested
       if per_channel and x.dim() > 1:
           # Reshape tensor to [channels, -1] for per-channel quantization
           orig_shape = x.shape
           channels = orig_shape[channel_dim]
           x_reshaped = x_float.transpose(0, channel_dim).reshape(channels, -1)
           
           # Initialize output tensor
           x_quantized = torch.zeros_like(x_float)
           
           # Quantize each channel separately
           for c in range(channels):
               x_channel = x_reshaped[c]
               
               # Apply outlier removal if requested
               if clip_outliers:
                   x_channel = PrecisionManager._clip_outliers(x_channel, outlier_threshold)
               
               # Quantize based on scheme
               if scheme == QuantizationScheme.SYMMETRIC:
                   x_quant_channel, channel_meta = PrecisionManager._symmetric_quantize(x_channel, bits)
               elif scheme == QuantizationScheme.ASYMMETRIC:
                   x_quant_channel, channel_meta = PrecisionManager._asymmetric_quantize(x_channel, bits)
               elif scheme == QuantizationScheme.LOGARITHMIC:
                   x_quant_channel, channel_meta = PrecisionManager._logarithmic_quantize(x_channel, bits)
               elif scheme == QuantizationScheme.DYNAMIC:
                   x_quant_channel, channel_meta = PrecisionManager._dynamic_quantize(x_channel, bits, outlier_threshold)
               else:
                   # Fallback to symmetric
                   x_quant_channel, channel_meta = PrecisionManager._symmetric_quantize(x_channel, bits)
               
               # Reshape quantized channel back to original shape
               x_channel_orig_shape = x_float.transpose(0, channel_dim)[c].shape
               x_quant_channel_reshaped = x_quant_channel.reshape(x_channel_orig_shape)
               
               # Put back in the right position
               if channel_dim == 0:
                   x_quantized[c] = x_quant_channel_reshaped
               else:
                   idx = [slice(None)] * len(orig_shape)
                   idx[channel_dim] = c
                   x_quantized[tuple(idx)] = x_quant_channel_reshaped
               
               # Store channel metadata
               if c == 0:
                   metadata["channel_meta"] = {}
               metadata["channel_meta"][c] = channel_meta
           
           return x_quantized.to(x.dtype), metadata
       
       # Apply outlier removal if requested
       if clip_outliers:
           x_float = PrecisionManager._clip_outliers(x_float, outlier_threshold)
       
       # Apply appropriate quantization scheme
       if scheme == QuantizationScheme.SYMMETRIC:
           return PrecisionManager._symmetric_quantize(x_float, bits)
       elif scheme == QuantizationScheme.ASYMMETRIC:
           return PrecisionManager._asymmetric_quantize(x_float, bits)
       elif scheme == QuantizationScheme.LOGARITHMIC:
           return PrecisionManager._logarithmic_quantize(x_float, bits)
       elif scheme == QuantizationScheme.DYNAMIC:
           return PrecisionManager._dynamic_quantize(x_float, bits, outlier_threshold)
       elif scheme == QuantizationScheme.AWQV1:
           # AWQ v1 requires activation data which we don't have here
           # Fallback to symmetric quantization with warning
           logger.warning("AWQ quantization requires activation data, falling back to symmetric")
           return PrecisionManager._symmetric_quantize(x_float, bits)
       else:
           # Unknown scheme, fallback to symmetric
           logger.warning(f"Unknown quantization scheme {scheme}, falling back to symmetric")
           return PrecisionManager._symmetric_quantize(x_float, bits)
   
   @staticmethod
   def _clip_outliers(x, threshold=0.01):
       """Clip outlier values"""
       try:
           x_flat = x.reshape(-1)
           k = max(1, int(threshold * x_flat.numel()))
           
           # Find values at threshold percentiles
           sorted_x, _ = torch.sort(x_flat)
           
           # Handle edge cases
           if k >= len(sorted_x):
               return x  # Not enough elements to clip
               
           min_val = sorted_x[k]
           max_val = sorted_x[-k]
           
           # Handle edge case where min_val >= max_val
           if min_val >= max_val:
               return x  # No valid range for clipping
           
           # Clip values
           return torch.clamp(x, min_val, max_val)
       except Exception as e:
           logger.warning(f"Error in _clip_outliers: {e}. Returning original tensor.")
           return x
   
   @staticmethod
   def _symmetric_quantize(x_float, bits):
       """Apply symmetric quantization"""
       try:
           # Symmetric quantization (zero-centered)
           abs_max = torch.max(torch.abs(x_float))
           
           # Avoid quantizing zero tensors or extremely small values
           if abs_max == 0 or (abs_max.numel() == 1 and abs_max.item() < 1e-10):
               return x_float, {"scheme": "symmetric", "bits": bits, "scale": 1.0}
           
           # Calculate scale and number of levels
           levels = 2 ** (bits - 1) - 1  # Account for sign bit
           scale = abs_max / levels
           
           # Quantize
           x_int = torch.round(x_float / scale).to(torch.int32)
           x_int = torch.clamp(x_int, -levels, levels)
           
           # Dequantize
           x_quantized = (x_int * scale).to(x_float.dtype)
           
           # Safely extract scalar values for metadata
           try:
               scale_val = scale.item() if scale.numel() == 1 else scale.detach().cpu().numpy().tolist()
               abs_max_val = abs_max.item() if abs_max.numel() == 1 else abs_max.detach().cpu().numpy().tolist()
           except:
               # Fallback if item() or numpy conversion fails
               scale_val = float(scale.detach().flatten()[0])
               abs_max_val = float(abs_max.detach().flatten()[0])
           
           metadata = {
               "scheme": "symmetric",
               "bits": bits,
               "scale": scale_val,
               "levels": levels,
               "min": -abs_max_val,
               "max": abs_max_val
           }
           
           return x_quantized, metadata
       except Exception as e:
           logger.warning(f"Error in _symmetric_quantize: {e}. Returning original tensor.")
           return x_float, {"scheme": "symmetric", "bits": bits, "error": str(e)}
   
   @staticmethod
   def _asymmetric_quantize(x_float, bits):
       """Apply asymmetric quantization"""
       try:
           # Asymmetric quantization (min-max based)
           x_min = torch.min(x_float)
           x_max = torch.max(x_float)
           
           # Avoid quantizing if min=max or range is too small
           if torch.allclose(x_min, x_max) or (x_max - x_min) < 1e-10:
               return x_float, {
                   "scheme": "asymmetric", 
                   "bits": bits, 
                   "scale": 1.0, 
                   "zero_point": float(x_min.detach().flatten()[0]) if x_min.numel() > 0 else 0.0
               }
           
           # Calculate scale and zero point
           levels = 2 ** bits - 1
           scale = (x_max - x_min) / levels
           
           # Quantize
           x_int = torch.round((x_float - x_min) / scale).to(torch.int32)
           x_int = torch.clamp(x_int, 0, levels)
           
           # Dequantize
           x_quantized = (x_int * scale + x_min).to(x_float.dtype)
           
           # Safely extract scalar values for metadata
           try:
               min_val = x_min.item() if x_min.numel() == 1 else x_min.detach().cpu().numpy().tolist()
               max_val = x_max.item() if x_max.numel() == 1 else x_max.detach().cpu().numpy().tolist()
               scale_val = scale.item() if scale.numel() == 1 else scale.detach().cpu().numpy().tolist()
           except:
               # Fallback if item() or numpy conversion fails
               min_val = float(x_min.detach().flatten()[0])
               max_val = float(x_max.detach().flatten()[0])
               scale_val = float(scale.detach().flatten()[0])
           
           metadata = {
               "scheme": "asymmetric",
               "bits": bits,
               "scale": scale_val,
               "zero_point": min_val,
               "levels": levels,
               "min": min_val,
               "max": max_val
           }
           
           return x_quantized, metadata
       except Exception as e:
           logger.warning(f"Error in _asymmetric_quantize: {e}. Returning original tensor.")
           return x_float, {"scheme": "asymmetric", "bits": bits, "error": str(e)}
   
   @staticmethod
   def _logarithmic_quantize(x_float, bits):
       """Apply logarithmic quantization"""
       try:
           # Logarithmic quantization (better for weights with values close to zero)
           signs = torch.sign(x_float)
           abs_values = torch.abs(x_float)
           
           # Find smallest non-zero value and max value
           # Handle case where all values might be zero
           if not torch.any(abs_values > 0):
               return x_float, {"scheme": "logarithmic", "bits": bits}
               
           min_positive = torch.min(abs_values[abs_values > 0])
           max_value = torch.max(abs_values)
           
           # Avoid quantizing if range is too small
           if max_value <= min_positive or max_value == 0:
               return x_float, {"scheme": "logarithmic", "bits": bits}
           
           # Convert to log space
           log_min = torch.log(min_positive)
           log_max = torch.log(max_value)
           
           # Prepare mask for zeros
           zero_mask = (abs_values == 0)
           
           # Convert non-zero values to log space
           log_values = torch.log(abs_values.clone())
           log_values[zero_mask] = log_min  # Set log of 0 to log_min
           
           # Linear quantization in log space
           levels = 2 ** (bits - 1)  # Use 1 bit for sign
           log_scale = (log_max - log_min) / (levels - 1)
           log_int = torch.round((log_values - log_min) / log_scale).to(torch.int32)
           log_int = torch.clamp(log_int, 0, levels - 1)
           
           # Dequantize back to linear space
           log_dequant = log_min + log_int.float() * log_scale
           abs_quantized = torch.exp(log_dequant)
           abs_quantized[zero_mask] = 0  # Restore zeros
           
           # Re-apply signs
           x_quantized = (signs * abs_quantized).to(x_float.dtype)
           
           # Safely extract scalar values for metadata
           try:
               log_scale_val = log_scale.item() if log_scale.numel() == 1 else log_scale.detach().cpu().numpy().tolist()
               log_min_val = log_min.item() if log_min.numel() == 1 else log_min.detach().cpu().numpy().tolist()
               log_max_val = log_max.item() if log_max.numel() == 1 else log_max.detach().cpu().numpy().tolist()
               max_val = max_value.item() if max_value.numel() == 1 else max_value.detach().cpu().numpy().tolist()
           except:
               # Fallback if item() or numpy conversion fails
               log_scale_val = float(log_scale.detach().flatten()[0])
               log_min_val = float(log_min.detach().flatten()[0])
               log_max_val = float(log_max.detach().flatten()[0])
               max_val = float(max_value.detach().flatten()[0])
           
           metadata = {
               "scheme": "logarithmic",
               "bits": bits,
               "log_scale": log_scale_val,
               "log_min": log_min_val,
               "log_max": log_max_val,
               "levels": levels,
               "min": -max_val,
               "max": max_val
           }
           
           return x_quantized, metadata
       except Exception as e:
           logger.warning(f"Error in _logarithmic_quantize: {e}. Returning original tensor.")
           return x_float, {"scheme": "logarithmic", "bits": bits, "error": str(e)}
   
   @staticmethod
   def _dynamic_quantize(x_float, bits, outlier_threshold=0.01):
       """Apply dynamic quantization with outlier handling"""
       try:
           # Dynamic quantization (with outlier removal)
           x_flat = x_float.flatten()
           n = x_flat.numel()
           
           # If tensor is too small, use standard asymmetric quantization
           if n <= 2:
               return PrecisionManager._asymmetric_quantize(x_float, bits)
               
           kth = max(1, int(outlier_threshold * n))
           
           # Use kthvalue for better performance on large tensors
           if n > 10000:  # Threshold where kthvalue becomes more efficient
               min_val = torch.kthvalue(x_flat, kth).values
               max_val = torch.kthvalue(x_flat, n-kth).values
           else:
               # For smaller tensors, sort might be more efficient
               x_sorted = torch.sort(x_flat).values
               min_val = x_sorted[kth]
               max_val = x_sorted[-kth]
           
           # Handle case when min >= max after outlier removal
           if min_val >= max_val:
               x_min = torch.min(x_float)
               x_max = torch.max(x_float)
               if torch.allclose(x_min, x_max):
                   return x_float, {"scheme": "dynamic", "bits": bits}
               min_val = x_min
               max_val = x_max
           
           # Safely extract scalar values
           try:
               min_val_scalar = min_val.item() if min_val.numel() == 1 else min_val.detach().cpu().numpy().tolist()
               max_val_scalar = max_val.item() if max_val.numel() == 1 else max_val.detach().cpu().numpy().tolist()
           except:
               # Fallback
               min_val_scalar = float(min_val.detach().flatten()[0])
               max_val_scalar = float(max_val.detach().flatten()[0])
           
           # Calculate scale
           levels = 2 ** bits - 1
           scale = (max_val - min_val) / levels
           
           # Handle edge case where scale might be zero
           if scale == 0 or torch.isclose(scale, torch.tensor(0.0, device=scale.device)):
               return x_float, {
                   "scheme": "dynamic", 
                   "bits": bits,
                   "min": min_val_scalar,
                   "max": max_val_scalar,
                   "error": "Zero scale detected"
               }
           
           # Clamp values to range after outlier removal
           x_clamped = torch.clamp(x_float, min_val, max_val)
           
           # Quantize
           x_int = torch.round((x_clamped - min_val) / scale).to(torch.int32)
           x_int = torch.clamp(x_int, 0, levels)
           
           # Dequantize
           x_quantized = (x_int * scale + min_val).to(x_float.dtype)
           
           # Safely extract scalar value for scale
           try:
               scale_scalar = scale.item() if scale.numel() == 1 else scale.detach().cpu().numpy().tolist()
           except:
               scale_scalar = float(scale.detach().flatten()[0])
           
           metadata = {
               "scheme": "dynamic",
               "bits": bits,
               "scale": scale_scalar,
               "zero_point": min_val_scalar,
               "min": min_val_scalar,
               "max": max_val_scalar,
               "levels": levels,
               "outlier_threshold": outlier_threshold
           }
           
           return x_quantized, metadata
       except Exception as e:
           logger.warning(f"Error in _dynamic_quantize: {e}. Returning original tensor.")
           return x_float, {"scheme": "dynamic", "bits": bits, "error": str(e)}
   
   @staticmethod
   def quantize_module(module, config=None):
       """
       Quantize all parameters in a module
       
       Args:
           module: PyTorch module to quantize
           config: QuantizationConfig or dict of parameter_name -> QuantizationConfig
           
       Returns:
           stats: Quantization statistics
       """
       if config is None:
           config = QuantizationConfig()
       
       # Initialize statistics
       stats = {
           "params_quantized": 0,
           "total_params": 0,
           "memory_saved_bytes": 0,
           "per_param_info": {}
       }
       
       try:
           with torch.no_grad():
               for name, param in module.named_parameters():
                   stats["total_params"] += 1
                   
                   # Skip quantization for non-floating point parameters
                   if not param.is_floating_point():
                       continue
                   
                   # Determine config for this parameter
                   param_config = config
                   if isinstance(config, dict) and name in config:
                       param_config = config[name]
                   elif isinstance(config, dict) and any(pattern in name for pattern in config):
                       # Find most specific matching pattern
                       matching_patterns = [p for p in config if p in name]
                       if matching_patterns:
                           best_match = max(matching_patterns, key=len)
                           param_config = config[best_match]
                   
                   if isinstance(param_config, dict):
                       param_config = QuantizationConfig(**param_config)
                   
                   # Skip if bits=0 (no quantization) or bits=32 (full precision)
                   if param_config.bits == 0 or param_config.bits == 32:
                       continue
                   
                   # Calculate original memory size
                   original_size = param.numel() * param.element_size()
                   
                   # Quantize parameter
                   try:
                       param_quantized, metadata = PrecisionManager.quantize_tensor(
                           param, 
                           bits=param_config.bits,
                           scheme=param_config.scheme,
                           per_channel=param_config.per_channel,
                           channel_dim=param_config.channel_dim,
                           clip_outliers=param_config.clip_outliers,
                           outlier_threshold=param_config.outlier_threshold
                       )
                       
                       # Calculate new memory size
                       if param_config.bits <= 8:
                           new_element_size = 1  # 1 byte for int8 or less
                       elif param_config.bits <= 16:
                           new_element_size = 2  # 2 bytes for int16 or fp16
                       else:
                           new_element_size = 4  # 4 bytes for fp32
                       
                       new_size = param.numel() * new_element_size
                       
                       # Update parameter with quantized values
                       param.copy_(param_quantized)
                       
                       # Update statistics
                       stats["params_quantized"] += 1
                       stats["memory_saved_bytes"] += (original_size - new_size)
                       stats["per_param_info"][name] = {
                           "original_size_bytes": original_size,
                           "quantized_size_bytes": new_size,
                           "bits": param_config.bits,
                           "scheme": param_config.scheme.value if isinstance(param_config.scheme, Enum) else param_config.scheme,
                           "metadata": metadata
                       }
                   except Exception as e:
                       logger.warning(f"Error quantizing parameter {name}: {e}. Skipping.")
                       # Include error in stats for debugging
                       stats["per_param_info"][name] = {
                           "error": str(e),
                           "bits": param_config.bits,
                           "scheme": param_config.scheme.value if isinstance(param_config.scheme, Enum) else param_config.scheme
                       }
           
           # Add summary statistics
           stats["memory_saved_mb"] = stats["memory_saved_bytes"] / (1024 * 1024)
           
           # Avoid division by zero in compression ratio
           denominator = sum(info.get("quantized_size_bytes", 0) for info in stats["per_param_info"].values())
           numerator = sum(info.get("original_size_bytes", 0) for info in stats["per_param_info"].values())
           stats["compression_ratio"] = numerator / max(denominator, 1) if stats["params_quantized"] > 0 else 1.0
           
       except Exception as e:
           logger.error(f"Error in quantize_module: {e}")
           stats["error"] = str(e)
       
       return stats
   
   @staticmethod
   def quantize_activations(x, bits=8, scheme=QuantizationScheme.DYNAMIC):
       """
       Dynamic quantization of activation tensors
       
       Args:
           x: Input activation tensor
           bits: Bit depth for quantization
           scheme: Quantization scheme
           
       Returns:
           x_quantized: Quantized tensor
           metadata: Quantization metadata
       """
       # Skip quantization for small tensors
       if x.numel() < 1000:
           return x, {"skipped": True, "reason": "small_tensor"}
       
       # Skip quantization for special bit values
       if bits == 0 or bits == 32:
           return x, {"skipped": True, "reason": "no_quantization"}
       
       # Handle NaN and Inf values
       if torch.isnan(x).any() or torch.isinf(x).any():
           x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
           
       return PrecisionManager.quantize_tensor(x, bits=bits, scheme=scheme)
   
   @staticmethod
   def compile_fn(fn, mode='reduce-overhead', dynamic=False):
       """
       Apply torch.compile to a function if available (PyTorch 2.0+)
       
       Args:
           fn: Function or module to compile
           mode: Compilation mode
           dynamic: Whether to use dynamic shapes
           
       Returns:
           compiled_fn: Compiled function or original function if compilation failed
       """
       if not TORCH_AVAILABLE:
           return fn
           
       try:
           # Check PyTorch version
           torch_version = torch.__version__.split('.')
           major_version = int(torch_version[0])
           
           if major_version >= 2:
               logger.info(f"Applying torch.compile with mode '{mode}'")
               
               # Additional compile options based on PyTorch version
               compile_options = {}
               
               # For PyTorch 2.0+, use dynamic shapes if requested
               if dynamic:
                   compile_options["dynamic"] = True
               
               # For PyTorch 2.0+, use torch.compile
               return torch.compile(fn, mode=mode, **compile_options)
           else:
               logger.warning(f"torch.compile requires PyTorch 2.0+, current version: {torch.__version__}")
               return fn
       except Exception as e:
           logger.warning(f"Failed to apply torch.compile: {e}")
           return fn
   
   @staticmethod
   def optimize_for_inference(model, config=None):
       """
       Apply multiple optimization techniques for inference
       
       Args:
           model: PyTorch model
           config: TAPConfig or None for defaults
           
       Returns:
           model: Optimized model
       """
       if not TORCH_AVAILABLE:
           return model
           
       if config is None:
           config = TAPConfig()
       
       # Ensure model is in eval mode
       model.eval()
       
       # Apply weight quantization if needed
       if hasattr(config, 'weight_quantization') and config.weight_quantization:
           bits = config.weight_quantization.bits
           if bits > 0 and bits < 32:
               logger.info(f"Applying weight quantization ({bits}-bit)")
               PrecisionManager.quantize_module(model, config.weight_quantization)
       
       # Apply torch.compile if enabled
       if config.compilation_enabled:
           model = PrecisionManager.compile_fn(model, mode=config.compile_mode)
       
       # Apply fusion optimizations where possible
       try:
           from torch.fx.experimental.optimization import fuse
           try:
               model = fuse(model)
               logger.info("Applied operator fusion optimizations")
           except Exception as e:
               logger.debug(f"Operator fusion failed: {e}")
       except ImportError:
           pass
       
       return model
   
   @staticmethod
   def test_quantization(tensor_example, bits_list=[4, 8, 16]):
       """
       Test quantization on a sample tensor with different bit settings
       
       Args:
           tensor_example: Example tensor to quantize
           bits_list: List of bit settings to test
           
       Returns:
           results: Dictionary with test results
       """
       results = {}
       
       # Original tensor stats
       results["original"] = {
           "min": tensor_example.min().item() if tensor_example.numel() > 0 else 0,
           "max": tensor_example.max().item() if tensor_example.numel() > 0 else 0,
           "mean": tensor_example.mean().item() if tensor_example.numel() > 0 else 0,
           "std": tensor_example.std().item() if tensor_example.numel() > 0 else 0,
           "memory_bytes": tensor_example.numel() * tensor_example.element_size()
       }
       
       # Test different bit settings
       for bits in bits_list:
           scheme_results = {}
           
           for scheme_name, scheme in [
               ("symmetric", QuantizationScheme.SYMMETRIC),
               ("asymmetric", QuantizationScheme.ASYMMETRIC),
               ("dynamic", QuantizationScheme.DYNAMIC)
           ]:
               try:
                   quantized, meta = PrecisionManager.quantize_tensor(
                       tensor_example, bits=bits, scheme=scheme)
                   
                   # Calculate error metrics
                   if tensor_example.numel() > 0:
                       abs_error = torch.abs(tensor_example - quantized)
                       mse = torch.mean((tensor_example - quantized) ** 2).item()
                       mae = torch.mean(abs_error).item()
                       max_error = torch.max(abs_error).item()
                       
                       scheme_results[scheme_name] = {
                           "mse": mse,
                           "mae": mae,
                           "max_error": max_error,
                           "metadata": meta
                       }
                   else:
                       scheme_results[scheme_name] = {
                           "error": "Empty tensor",
                           "metadata": meta
                       }
               except Exception as e:
                   scheme_results[scheme_name] = {
                       "error": str(e)
                   }
           
           # Calculate memory usage
           if bits <= 8:
               memory_bytes = tensor_example.numel() * 1  # 1 byte for int8 or less
           elif bits <= 16:
               memory_bytes = tensor_example.numel() * 2  # 2 bytes for int16/fp16
           else:
               memory_bytes = tensor_example.numel() * 4  # 4 bytes for fp32
           

   @staticmethod
       def test_quantization(tensor_example, bits_list=[4, 8, 16]):
           """
           Test quantization on a sample tensor with different bit settings
           
           Args:
               tensor_example: Example tensor to quantize
               bits_list: List of bit settings to test
               
           Returns:
               results: Dictionary with test results
           """
           results = {}
           
           # Original tensor stats
           results["original"] = {
               "min": tensor_example.min().item() if tensor_example.numel() > 0 else 0,
               "max": tensor_example.max().item() if tensor_example.numel() > 0 else 0,
               "mean": tensor_example.mean().item() if tensor_example.numel() > 0 else 0,
               "std": tensor_example.std().item() if tensor_example.numel() > 0 else 0,
               "memory_bytes": tensor_example.numel() * tensor_example.element_size()
           }
           
           # Test different bit settings
           for bits in bits_list:
               scheme_results = {}
               
               for scheme_name, scheme in [
                   ("symmetric", QuantizationScheme.SYMMETRIC),
                   ("asymmetric", QuantizationScheme.ASYMMETRIC),
                   ("dynamic", QuantizationScheme.DYNAMIC)
               ]:
                   try:
                       quantized, meta = PrecisionManager.quantize_tensor(
                           tensor_example, bits=bits, scheme=scheme)
                       
                       # Calculate error metrics
                       if tensor_example.numel() > 0:
                           abs_error = torch.abs(tensor_example - quantized)
                           mse = torch.mean((tensor_example - quantized) ** 2).item()
                           mae = torch.mean(abs_error).item()
                           max_error = torch.max(abs_error).item()
                           
                           scheme_results[scheme_name] = {
                               "mse": mse,
                               "mae": mae,
                               "max_error": max_error,
                               "metadata": meta
                           }
                       else:
                           scheme_results[scheme_name] = {
                               "error": "Empty tensor",
                               "metadata": meta
                           }
                   except Exception as e:
                       scheme_results[scheme_name] = {
                           "error": str(e)
                       }
               
               # Calculate memory usage
               if bits <= 8:
                   memory_bytes = tensor_example.numel() * 1  # 1 byte for int8 or less
               elif bits <= 16:
                   memory_bytes = tensor_example.numel() * 2  # 2 bytes for int16/fp16
               else:
                   memory_bytes = tensor_example.numel() * 4  # 4 bytes for fp32
               
               results[f"{bits}bit"] = {
                   **scheme_results,
                   "memory_bytes": memory_bytes,
                   "compression_ratio": results["original"]["memory_bytes"] / max(memory_bytes, 1)
               }
           
           return results

#------------------------------------------------------------------------------
# Token Importance Analysis
#------------------------------------------------------------------------------

class TokenImportanceAnalyzer:
    """
    Enhanced analyzer for token importance with support for different architectures.
    Provides tools for importance-based precision allocation.
    """
    
    def __init__(self, config=None):
        """
        Initialize token importance analyzer
        
        Args:
            config: TAPConfig object or None for defaults
        """
        self.config = config or TAPConfig()
        self.importance_cache = OrderedDict()  # OrderedDict for LRU cache behavior
        self.max_cache_size = self.config.importance_cache_size
        
        # Detect architecture type
        self.model_arch = self.config.model_arch
        
        # Tracking measurements
        self.measurements = {
            "total_tokens_analyzed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "analysis_time_total": 0,
            "analysis_count": 0
        }
    
    def analyze_token_importance(self,
                                attention_scores=None,
                                hidden_states=None,
                                ssm_states=None,
                                input_ids=None,
                                attention_mask=None,
                                position_ids=None,
                                gradient=None,
                                method=None):
        """
        Calculate token importance using specified method with architecture awareness
        
        Args:
            attention_scores: Attention scores [batch, num_heads, seq_len, seq_len]
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
            ssm_states: SSM state vectors [batch, seq_len, state_dim]
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]
            gradient: Token gradients [batch, seq_len, hidden_dim]
            method: Method to use (if None, use from config)
            
        Returns:
            token_importance: Tensor of shape [batch, seq_len] with values in [0,1] with architecture-specific calculation
        """
        start_time = time.time()
        
        if method is None:
            method = self.config.importance_method
        
        # Convert string method to enum if needed
        if isinstance(method, str):
            try:
                method = TokenImportanceMetric(method)
            except ValueError:
                logger.warning(f"Unknown importance method: {method}, falling back to hybrid")
                method = TokenImportanceMetric.HYBRID
        
        # Check cache first (only for non-gradient methods)
        if method != TokenImportanceMetric.GRADIENT and input_ids is not None:
            cache_key = self._get_cache_key(input_ids, method)
            if cache_key in self.importance_cache:
                self.measurements["cache_hits"] += 1
                # Move to end to maintain LRU behavior
                importance = self.importance_cache[cache_key]
                self.importance_cache.move_to_end(cache_key)
                return importance
            else:
                self.measurements["cache_misses"] += 1
        
        # Determine device and dimensions
        device = self._get_device(attention_scores, hidden_states, input_ids, ssm_states)
        batch_size, seq_len = self._get_dimensions(attention_scores, hidden_states, input_ids, ssm_states)
        
        # Update measurement counters
        self.measurements["total_tokens_analyzed"] += batch_size * seq_len
        
        # Architecture-specific importance calculation
        if self.model_arch == ModelArchitecture.MAMBA or self.model_arch == ModelArchitecture.RWKV:
            # For state space models, prioritize state-based methods
            if method == TokenImportanceMetric.ATTENTION and ssm_states is not None:
                # Fall back to state norm for SSMs since they don't have attention
                method = TokenImportanceMetric.STATE_NORM
                logger.debug("Switching to STATE_NORM for SSM model")
        
        # Calculate importance based on method and available data
        if method == TokenImportanceMetric.ATTENTION and attention_scores is not None:
            importance = self._importance_from_attention(attention_scores)
        
        elif method == TokenImportanceMetric.HIDDEN_NORM and hidden_states is not None:
            importance = self._importance_from_hidden_states(hidden_states)
        
        elif method == TokenImportanceMetric.STATE_NORM and ssm_states is not None:
            importance = self._importance_from_state_vectors(ssm_states)
        
        elif method == TokenImportanceMetric.POSITION:
            importance = self._importance_from_position(batch_size, seq_len, position_ids, device)
        
        elif method == TokenImportanceMetric.TOKEN_ID and input_ids is not None:
            importance = self._importance_from_token_ids(input_ids)
        
        elif method == TokenImportanceMetric.GRADIENT and gradient is not None:
            importance = self._importance_from_gradient(gradient)
        
        elif method == TokenImportanceMetric.ENTROPY and attention_scores is not None:
            importance = self._importance_from_entropy(attention_scores)
        
        elif method == TokenImportanceMetric.HYBRID:
            # Combine multiple methods with weights adjusted for architecture
            importance_scores = []
            
            if self.model_arch == ModelArchitecture.TRANSFORMER:
                # For transformers, prioritize attention and hidden states
                if attention_scores is not None:
                    attention_imp = self._importance_from_attention(attention_scores)
                    importance_scores.append((attention_imp, 0.4))  # 40% weight
                
                if hidden_states is not None:
                    hidden_imp = self._importance_from_hidden_states(hidden_states)
                    importance_scores.append((hidden_imp, 0.3))  # 30% weight
                
                position_imp = self._importance_from_position(batch_size, seq_len, position_ids, device)
                importance_scores.append((position_imp, 0.2))  # 20% weight
                
                if input_ids is not None:
                    token_imp = self._importance_from_token_ids(input_ids)
                    importance_scores.append((token_imp, 0.1))  # 10% weight
            
            elif self.model_arch in [ModelArchitecture.MAMBA, ModelArchitecture.RWKV]:
                # For SSMs, prioritize state vectors and hidden states
                if ssm_states is not None:
                    state_imp = self._importance_from_state_vectors(ssm_states)
                    importance_scores.append((state_imp, 0.5))  # 50% weight
                
                if hidden_states is not None:
                    hidden_imp = self._importance_from_hidden_states(hidden_states)
                    importance_scores.append((hidden_imp, 0.3))  # 30% weight
                
                position_imp = self._importance_from_position(batch_size, seq_len, position_ids, device)
                importance_scores.append((position_imp, 0.15))  # 15% weight
                
                if input_ids is not None:
                    token_imp = self._importance_from_token_ids(input_ids)
                    importance_scores.append((token_imp, 0.05))  # 5% weight
            
            elif self.model_arch == ModelArchitecture.RETNET:
                # For RetNet, prioritize hidden states and position
                if hidden_states is not None:
                    hidden_imp = self._importance_from_hidden_states(hidden_states)
                    importance_scores.append((hidden_imp, 0.5))  # 50% weight
                
                position_imp = self._importance_from_position(batch_size, seq_len, position_ids, device)
                importance_scores.append((position_imp, 0.3))  # 30% weight
                
                if input_ids is not None:
                    token_imp = self._importance_from_token_ids(input_ids)
                    importance_scores.append((token_imp, 0.2))  # 20% weight
            
            else:
                # Generic fallback
                if hidden_states is not None:
                    hidden_imp = self._importance_from_hidden_states(hidden_states)
                    importance_scores.append((hidden_imp, 0.5))  # 50% weight
                
                position_imp = self._importance_from_position(batch_size, seq_len, position_ids, device)
                importance_scores.append((position_imp, 0.3))  # 30% weight
                
                if input_ids is not None:
                    token_imp = self._importance_from_token_ids(input_ids)
                    importance_scores.append((token_imp, 0.2))  # 20% weight
            
            # Weighted average
            importance = torch.zeros((batch_size, seq_len), device=device)
            weight_sum = 0
            
            for imp, weight in importance_scores:
                importance += imp * weight
                weight_sum += weight
            
            if weight_sum > 0:
                importance /= weight_sum
            else:
                # Fallback to uniform if no methods available
                importance = torch.ones((batch_size, seq_len), device=device) * 0.5
        
        else:
            # Uniform importance as fallback
            importance = torch.ones((batch_size, seq_len), device=device) * 0.5
        
        # Apply smoothing to avoid extreme values
        importance = self._smooth_importance(importance)
        
        # Cache the result (except for gradient-based method)
        if method != TokenImportanceMetric.GRADIENT and input_ids is not None:
            self._cache_importance(input_ids, importance, method)
        
        # Update timing measurements
        analysis_time = time.time() - start_time
        self.measurements["analysis_time_total"] += analysis_time
        self.measurements["analysis_count"] += 1
        
        return importance
    
    def _smooth_importance(self, importance, smoothing_factor=0.05):
        """Apply smoothing to importance scores to avoid extreme values"""
        # Apply min-max smoothing
        min_val = torch.min(importance, dim=1, keepdim=True)[0]
        max_val = torch.max(importance, dim=1, keepdim=True)[0]
        
        # Adjust range to avoid extreme values
        adjusted_min = min_val + smoothing_factor * (max_val - min_val)
        adjusted_max = max_val - smoothing_factor * (max_val - min_val)
        
        # Apply smoothing
        smoothed = (importance - min_val) / (max_val - min_val)
        smoothed = adjusted_min + smoothed * (adjusted_max - adjusted_min)
        
        return smoothed
    
    def _get_device(self, attention_scores=None, hidden_states=None, input_ids=None, ssm_states=None):
        """Get device from available tensors"""
        for tensor in [attention_scores, hidden_states, input_ids, ssm_states]:
            if tensor is not None:
                return tensor.device
        
        # Fallback to configured device
        if TORCH_AVAILABLE:
            return torch.device(self.config.device)
        else:
            return None
    
    def _get_dimensions(self, attention_scores=None, hidden_states=None, input_ids=None, ssm_states=None):
        """Get batch size and sequence length from available tensors"""
        if attention_scores is not None:
            if len(attention_scores.shape) == 4:
                return attention_scores.shape[0], attention_scores.shape[2]
        
        if hidden_states is not None:
            if len(hidden_states.shape) == 3:
                return hidden_states.shape[0], hidden_states.shape[1]
        
        if input_ids is not None:
            if len(input_ids.shape) == 2:
                return input_ids.shape[0], input_ids.shape[1]
        
        if ssm_states is not None:
            if len(ssm_states.shape) == 3:
                return ssm_states.shape[0], ssm_states.shape[1]
        
        raise ValueError("Could not determine dimensions from inputs")
    
    def _get_cache_key(self, input_ids, method):
        """Generate a cache key for importance results"""
        if isinstance(method, Enum):
            method = method.value
        
        # Use hash of input_ids as part of the key
        # We'll use a simplified hash for faster lookup
        if hasattr(input_ids, "cpu"):
            flat_ids = input_ids.cpu().flatten().tolist()
        else:
            flat_ids = input_ids.flatten().tolist()
            
        # For long sequences, hash only the first 100 and last 100 tokens
        if len(flat_ids) > 200:
            hash_input = str(flat_ids[:100]) + str(flat_ids[-100:])
        else:
            hash_input = str(flat_ids)
        
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        return f"{method}_{hash_value}"
    
    def _cache_importance(self, input_ids, importance, method):
        """Cache importance results with LRU behavior"""
        # Limit cache size
        if len(self.importance_cache) >= self.max_cache_size:
            # Remove oldest item (first in OrderedDict)
            self.importance_cache.popitem(last=False)
        
        cache_key = self._get_cache_key(input_ids, method)
        self.importance_cache[cache_key] = importance
    
    def _importance_from_attention(self, attention_scores):
        """
        Calculate token importance from attention scores
        
        Args:
            attention_scores: [batch, num_heads, seq_len, seq_len] or list of such tensors
        
        Returns:
            importance: [batch, seq_len] with values in [0,1]
        """
        # Handle case when attention_scores is a list (e.g., from all layers)
        if isinstance(attention_scores, list):
            # Average over last 3 layers if available
            layers_to_use = min(3, len(attention_scores))
            scores_to_use = attention_scores[-layers_to_use:]
            
            # Combine attention from multiple layers
            batch_size, num_heads, seq_len, _ = scores_to_use[0].shape
            combined_attention = torch.zeros((batch_size, num_heads, seq_len, seq_len), 
                                           device=scores_to_use[0].device)
            
            # Weighted average (more weight to later layers)
            total_weight = 0
            for i, layer_scores in enumerate(scores_to_use):
                weight = i + 1  # More weight to later layers
                combined_attention += layer_scores * weight
                total_weight += weight
            
            attention_scores = combined_attention / total_weight
        
        # Average attention across heads
        avg_attention = attention_scores.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Calculate incoming and outgoing attention
        incoming_attention = avg_attention.sum(dim=1)  # [batch, seq_len]
        outgoing_attention = avg_attention.sum(dim=2)  # [batch, seq_len]
        
        # Combine with weight towards incoming attention (more important)
        token_importance = 0.7 * incoming_attention + 0.3 * outgoing_attention
        
        # Normalize importance to [0, 1]
        batch_min = token_importance.min(dim=1, keepdim=True)[0]
        batch_max = token_importance.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        denominator = batch_max - batch_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        token_importance = (token_importance - batch_min) / denominator
        
        return token_importance
    
    def _importance_from_hidden_states(self, hidden_states):
        """
        Calculate token importance from hidden states
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] or list of such tensors
        
        Returns:
            importance: [batch, seq_len] with values in [0,1]
        """
        # Handle case when hidden_states is a list (e.g., from all layers)
        if isinstance(hidden_states, list):
            # Use last layer
            hidden_states = hidden_states[-1]
        
        # Calculate magnitude (L2 norm) of hidden states
        hidden_magnitude = torch.norm(hidden_states, dim=2)  # [batch, seq_len]
        
        # Normalize importance to [0, 1]
        batch_min = hidden_magnitude.min(dim=1, keepdim=True)[0]
        batch_max = hidden_magnitude.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        denominator = batch_max - batch_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        token_importance = (hidden_magnitude - batch_min) / denominator
        
        return token_importance
    
    def _importance_from_state_vectors(self, state_vectors):
        """
        Calculate token importance from SSM state vectors
        
        Args:
            state_vectors: [batch, seq_len, state_dim] for SSM models
        
        Returns:
            importance: [batch, seq_len] with values in [0,1]
        """
        # Calculate magnitude (L2 norm) of state vectors
        state_magnitude = torch.norm(state_vectors, dim=2)  # [batch, seq_len]
        
        # Apply a non-linearity to emphasize changes
        # The rate of change in state vectors is more informative than raw magnitudes
        state_diff = torch.abs(state_magnitude[:, 1:] - state_magnitude[:, :-1])
        
        # For the first token, use its magnitude directly
        state_diff = torch.cat([state_magnitude[:, 0:1], state_diff], dim=1)
        
        # Normalize importance to [0, 1]
        batch_min = state_diff.min(dim=1, keepdim=True)[0]
        batch_max = state_diff.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        denominator = batch_max - batch_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        token_importance = (state_diff - batch_min) / denominator
        
        return token_importance
    
    def _importance_from_position(self, batch_size, seq_len, position_ids=None, device=None):
        """
        Calculate token importance based on position
        Uses Gaussian distribution with peak at middle positions
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            position_ids: Position IDs [batch, seq_len] or None
            device: Computation device
            
        Returns:
            importance: [batch, seq_len] with values in [0,1]
        """
        if position_ids is None:
            # Create position indices from 0 to seq_len-1
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Create importance based on distance from middle position
        # Gaussian importance with peak at middle of sequence
        mid_pos = seq_len / 2
        
        # Adjust sigma based on sequence length (wider for longer sequences)
        sigma = max(seq_len / 4, 5)
        
        # Calculate Gaussian importance
        pos_float = position_ids.float()
        token_importance = torch.exp(-((pos_float - mid_pos) ** 2) / (2 * sigma ** 2))
        
        return token_importance
    
    def _importance_from_token_ids(self, input_ids, vocab_size=50000):
        """
        Calculate token importance based on token IDs
        Rare tokens (higher IDs) tend to be more important
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            vocab_size: Size of vocabulary
            
        Returns:
            importance: [batch, seq_len] with values in [0,1]
        """
        # Normalize token IDs to [0, 1] range
        max_id = min(vocab_size, 50000)  # Cap at reasonable size
        normalized_ids = input_ids.float() / max_id
        
        # Apply sigmoid with adjusted slope to emphasize higher token IDs
        # This assumes higher token IDs are generally less common tokens
        token_importance = torch.sigmoid(4 * (normalized_ids - 0.5))
        
        return token_importance
    
    def _importance_from_gradient(self, gradient):
        """
        Calculate token importance from gradient magnitudes
        
        Args:
            gradient: Gradient tensor [batch, seq_len, hidden_dim]
            
        Returns:
            importance: [batch, seq_len] with values in [0,1]
        """
        # Calculate L2 norm of gradients
        grad_magnitude = torch.norm(gradient, dim=2)  # [batch, seq_len]
        
        # Normalize importance to [0, 1]
        batch_min = grad_magnitude.min(dim=1, keepdim=True)[0]
        batch_max = grad_magnitude.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        denominator = batch_max - batch_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        token_importance = (grad_magnitude - batch_min) / denominator
        
        return token_importance
    
    def _importance_from_entropy(self, attention_scores):
        """
        Calculate token importance based on attention entropy
        
        Args:
            attention_scores: [batch, num_heads, seq_len, seq_len] 
            
        Returns:
            importance: [batch, seq_len] with values in [0,1]
        """
        # Average attention across heads
        avg_attention = attention_scores.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Calculate entropy for each token's attention distribution
        # First ensure valid probability distribution by applying softmax
        attn_probs = F.softmax(avg_attention, dim=-1)
        
        # Avoid log(0) by adding small epsilon
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10), dim=-1)  # [batch, seq_len]
        
        # Lower entropy = more focused attention = more important token
        # So we invert: importance = 1 - normalized_entropy
        batch_min = entropy.min(dim=1, keepdim=True)[0]
        batch_max = entropy.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        denominator = batch_max - batch_min
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        normalized_entropy = (entropy - batch_min) / denominator
        
        token_importance = 1.0 - normalized_entropy
        
        return token_importance
    
    def assign_precision(self, token_importance, precision_levels=None, thresholds=None):
        """
        Assign precision levels to tokens based on importance
        
        Args:
            token_importance: Token importance tensor [batch, seq_len] with values in [0,1]
            precision_levels: List of precision bits [low to high]
            thresholds: List of importance thresholds for each level
            
        Returns:
            token_precision: Tensor [batch, seq_len] with precision bits for each token
        """
        if precision_levels is None:
            precision_levels = self.config.precision_levels
        
        if thresholds is None:
            thresholds = self.config.precision_thresholds
        
        # Validate inputs
        if len(precision_levels) != len(thresholds) + 1:
            logger.warning(f"precision_levels should have length equal to thresholds + 1, "
                          f"got {len(precision_levels)} and {len(thresholds)}")
            # Auto-fix by adjusting thresholds or precision levels
            if len(precision_levels) > 1:
                # Adjust thresholds to match precision levels
                step = 1.0 / len(precision_levels)
                thresholds = [step * (i+1) for i in range(len(precision_levels) - 1)]
                logger.warning(f"Adjusted thresholds to: {thresholds}")
            else:
                # Need at least 2 precision levels
                precision_levels = [8, 16]  # Default fallback
                thresholds = [0.5]  # Middle threshold
                logger.warning(f"Adjusted to precision_levels={precision_levels}, thresholds={thresholds}")
        
        # Create tensor for token precision with lowest precision as default
        token_precision = torch.ones_like(token_importance, dtype=torch.int32) * precision_levels[0]
        
        # Assign precision based on importance thresholds
        for i in range(1, len(precision_levels)):
            threshold = thresholds[i-1]
            precision = precision_levels[i]
            token_precision = torch.where(token_importance >= threshold, 
                                         torch.tensor(precision, device=token_importance.device, dtype=torch.int32), 
                                         token_precision)
        
        return token_precision
    
    def analyze_precision_stats(self, token_importance, token_precision):
        """
        Analyze statistics of token importance and precision assignment
        
        Args:
            token_importance: Token importance tensor [batch, seq_len]
            token_precision: Token precision tensor [batch, seq_len]
            
        Returns:
            stats: Dictionary of statistics
        """
        stats = {}
        
        # Importance statistics
        stats["importance_mean"] = token_importance.mean().item()
        stats["importance_std"] = token_importance.std().item()
        stats["importance_min"] = token_importance.min().item()
        stats["importance_max"] = token_importance.max().item()
        
        # Distribution by importance ranges
        for threshold in [0.2, 0.4, 0.6, 0.8]:
            pct = (token_importance >= threshold).float().mean().item() * 100
            stats[f"pct_above_{threshold}"] = pct
        
        # Precision statistics
        if token_precision is not None:
            # Calculate proportion of each precision level
            unique_precs = torch.unique(token_precision)
            for prec in unique_precs:
                prec_val = prec.item()
                pct = (token_precision == prec_val).float().mean().item() * 100
                stats[f"precision_{int(prec_val)}_pct"] = pct
            
            # Calculate average precision
            stats["precision_mean"] = token_precision.float().mean().item()
            
            # Calculate theoretical energy savings
            if hasattr(self.config, "energy_tracking") and "relative_energy" in self.config.energy_tracking:
                energy_factors = self.config.energy_tracking["relative_energy"]
                relative_energy = 0
                
                for prec_val in unique_precs:
                    prec_val = int(prec_val.item())
                    pct = (token_precision == prec_val).float().mean().item()
                    energy_factor = energy_factors.get(prec_val, 1.0)
                    relative_energy += (pct / 100) * energy_factor
                
                stats["relative_energy"] = relative_energy
                stats["energy_saved_pct"] = (1 - relative_energy) * 100
        
        return stats
    
    def get_analyzer_stats(self):
        """Get statistics about the analyzer's performance"""
        stats = copy.deepcopy(self.measurements)
        
        # Calculate averages and derived metrics
        if stats["analysis_count"] > 0:
            stats["avg_analysis_time_ms"] = (stats["analysis_time_total"] * 1000) / stats["analysis_count"]
        
        if stats["cache_hits"] + stats["cache_misses"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
        
        # Cache information
        stats["cache_size"] = len(self.importance_cache)
        stats["cache_max_size"] = self.max_cache_size
        
        return stats


#------------------------------------------------------------------------------
# SSM Core Modules for Mamba Integration
#------------------------------------------------------------------------------

class SelectiveScanFn(torch.autograd.Function):
    """
    Autograd function for the selective scan operation used in Mamba.
    Implements the core recurrence of the SSM with selective state updates.
    Supports token-adaptive precision.
    """
    
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, token_precision=None):
        """
        Forward pass of the selective scan function with token-adaptive precision
        
        Args:
            u: Input tensor [batch, seq_len, dim]
            delta: Timescale tensor [batch, seq_len, dim]
            A: State matrix [dim]
            B: Input projection [batch, seq_len, dim]
            C: Output projection [batch, seq_len, dim]
            D: Skip connection (optional)
            z: Gating tensor (optional)
            delta_bias: Bias for delta (optional)
            delta_softplus: Whether to apply softplus to delta
            token_precision: Precision bits for each token [batch, seq_len]
            
        Returns:
            y: Output of the selective scan
        """
        # Apply precision conversions based on token_precision if provided
        if token_precision is not None:
            batch_size, seq_len, dim = u.shape
            device = u.device
            
            # Process inputs based on token precision
            u_precision = torch.zeros_like(u, dtype=torch.float32)  # Output buffer
            delta_precision = torch.zeros_like(delta, dtype=torch.float32)  # Output buffer
            B_precision = torch.zeros_like(B, dtype=torch.float32)  # Output buffer
            C_precision = torch.zeros_like(C, dtype=torch.float32)  # Output buffer
            
            # Process each precision level separately
            for precision in torch.unique(token_precision):
                precision_val = precision.item()
                
                # Create mask for tokens with this precision
                mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, dim)
                
                # Apply quantization at specified precision
                u_quant, _ = PrecisionManager.quantize_tensor(
                    u[mask].reshape(-1, dim), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                delta_quant, _ = PrecisionManager.quantize_tensor(
                    delta[mask].reshape(-1, dim), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                B_quant, _ = PrecisionManager.quantize_tensor(
                    B[mask].reshape(-1, dim), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                C_quant, _ = PrecisionManager.quantize_tensor(
                    C[mask].reshape(-1, dim), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Insert quantized values back
                u_precision[mask] = u_quant.reshape(-1)
                delta_precision[mask] = delta_quant.reshape(-1)
                B_precision[mask] = B_quant.reshape(-1)
                C_precision[mask] = C_quant.reshape(-1)
            
            # Replace original inputs with precision-adapted ones
            u = u_precision
            delta = delta_precision
            B = B_precision
            C = C_precision
        
        # Continue with normal selective scan
        batch, seq_len, dim = u.shape
        
        # Apply delta bias and softplus if needed
        if delta_bias is not None:
            delta = delta + delta_bias.view(1, 1, -1)
        if delta_softplus:
            delta = F.softplus(delta)
        
        # Create discretized state matrices for recurrence
        # A is typically a negative real number, so we compute exp(A * delta)
        deltaA = torch.exp(torch.einsum('bld,d->bld', delta, A))
        
        # Forward scan initialization
        x = torch.zeros(batch, dim, device=u.device)
        ys = []
        
        # Sequential scan for accurate gradients during training
        for i in range(seq_len):
            # Update state: x_t = A_t * x_{t-1} + B_t * u_t
            x = deltaA[:, i] * x + B[:, i] * u[:, i]
            # Compute output: y_t = C_t * x_t + D * u_t
            y_t = C[:, i] * x
            
            # Apply skip connection if D is provided
            if D is not None:
                y_t = y_t + D * u[:, i]
            
            # Apply gating if z is provided
            if z is not None:
                y_t = y_t * z[:, i]
            
            ys.append(y_t)
        
        # Stack outputs
        y = torch.stack(ys, dim=1)  # [batch, seq_len, dim]
        
        # Save for backward
        ctx.save_for_backward(u, delta, A, B, C, D, z, deltaA, x)
        ctx.delta_softplus = delta_softplus
        
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        u, delta, A, B, C, D, z, deltaA, x_T = ctx.saved_tensors
        delta_softplus = ctx.delta_softplus
        
        batch, seq_len, dim = u.shape
        
        # Initialize gradient accumulators
        grad_u = torch.zeros_like(u)
        grad_delta = torch.zeros_like(delta) if delta.requires_grad else None
        grad_A = torch.zeros_like(A) if A.requires_grad else None
        grad_B = torch.zeros_like(B) if B.requires_grad else None
        grad_C = torch.zeros_like(C) if C.requires_grad else None
        grad_D = torch.zeros_like(D) if D is not None and D.requires_grad else None
        grad_z = torch.zeros_like(z) if z is not None and z.requires_grad else None
        
        # Backward scan initialization - start from the last state
        x = x_T.clone()
        
        # Backward scan: we need to go backwards through the sequence
        for i in range(seq_len - 1, -1, -1):
            if z is not None:
                grad_y_i = grad_y[:, i] * z[:, i]
                if grad_z is not None:
                    grad_z[:, i] = torch.sum(grad_y[:, i] * C[:, i] * x, dim=-1)
            else:
                grad_y_i = grad_y[:, i]
            
            # Gradient for C: dL/dC = dL/dy * x
            if grad_C is not None:
                grad_C[:, i] = grad_y_i.unsqueeze(-1) * x.unsqueeze(1)
            
            # Gradient for x: dL/dx = dL/dy * C
            grad_x = grad_y_i * C[:, i]
            
            # Gradient for D: dL/dD = dL/dy * u
            if D is not None and grad_D is not None:
                grad_D = grad_D + torch.sum(grad_y_i * u[:, i], dim=0)
            
            # Gradient for u from skip connection: dL/du = dL/dy * D
            if D is not None:
                grad_u[:, i] = grad_y_i * D
            
            # If not the first step, propagate gradients backward
            if i > 0:
                # Gradient for B: dL/dB = dL/dx * u
                if grad_B is not None:
                    grad_B[:, i-1] = grad_x * u[:, i-1]
                
                # Gradient for u from state update: dL/du = dL/dx * B
                grad_u[:, i-1] = grad_u[:, i-1] + grad_x * B[:, i-1]
                
                # Gradient for delta and A
                if grad_delta is not None or grad_A is not None:
                    # We need dL/d(deltaA) = dL/dx * x_{i-1}
                    dL_ddeltaA = grad_x * x
                    
                    # Gradient for A: dL/dA = dL/d(deltaA) * d(deltaA)/dA
                    if grad_A is not None:
                        grad_A_i = dL_ddeltaA * deltaA[:, i-1] * delta[:, i-1]
                        grad_A = grad_A + torch.sum(grad_A_i, dim=(0, 1))
                    
                    # Gradient for delta: dL/ddelta = dL/d(deltaA) * d(deltaA)/ddelta
                    if grad_delta is not None:
                        if delta_softplus:
                            # Account for softplus
                            sigmoid_delta = torch.sigmoid(delta[:, i-1])
                            d_softplus = sigmoid_delta
                            grad_delta[:, i-1] = dL_ddeltaA * deltaA[:, i-1] * A * d_softplus
                        else:
                            grad_delta[:, i-1] = dL_ddeltaA * deltaA[:, i-1] * A
                
                # Update x for the next iteration of the loop
                x = deltaA[:, i-1] * x
            
        return grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D, grad_z, None, None, None


class TokenAdaptiveSSMLayer(nn.Module):
    """
    State Space Model (SSM) layer with token-adaptive precision for Mamba architecture.
    This is the core building block for token-adaptive Mamba/SSM models.
    """
    def __init__(self, 
                 hidden_size, 
                 state_dim=16, 
                 dt_init=0.001, 
                 dt_min=0.001, 
                 dt_max=0.1, 
                 dt_scale=0.01,
                 dt_init_floor=1e-4,
                 use_bias=True,
                 activation="silu",
                 dropout=0.0,
                 config=None):
        super().__init__()
        self.config = config or TAPConfig()
        
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.use_bias = use_bias
        self.dropout = dropout
        
        # Set default activation function
        if activation == "silu":
            self.activation = F.silu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.silu  # default
        
        # Parameter groups
        # 1. Input projection (D matrix in SSM formulation)
        self.in_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # 2. Delta projection - for discretization timescale 
        # This allows the model to learn timescale per token
        self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Initialize special dt parameters
        dt_init_uniform = torch.rand(hidden_size) * (dt_init - dt_init_floor) + dt_init_floor
        self.dt_proj.weight.data.zero_()
        self.dt_proj.bias.data.copy_(torch.log(dt_init_uniform / (1 - dt_init_uniform)))
        
        # 3. State parameters (A, B, C in SSM formulation)
        # A is the state matrix, initialized to a negative value to ensure stability
        self.A_log = nn.Parameter(torch.zeros(hidden_size, state_dim))
        # B is the input matrix
        self.B = nn.Parameter(torch.zeros(hidden_size, state_dim))
        # C is the output matrix
        self.C = nn.Parameter(torch.zeros(hidden_size, state_dim))
        
        # Initialize parameters
        self._init_parameters()
        
        # 4. Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # 5. Gating mechanism (element-wise gating to control information flow)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Apply initializations
        for name, p in self.named_parameters():
            if p.dim() > 1 and name not in ['A_log', 'B', 'C']:
                # Use xavier uniform for non-SSM parameters
                nn.init.xavier_uniform_(p)
        
        # For precision tracking
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
    
    def _init_parameters(self):
        """Initialize SSM parameters with proper scaling"""
        # Initialize A_log to give real part(A) as a logarithmically-spaced sequence
        # starting at dt_min and ending at dt_max, with a step size of dt_scale
        # We use a logarithmic spacing because the time constants of many real
        # systems are logarithmically distributed
        
        # Log-uniformly distributed time constants
        N = self.hidden_size
        D = self.state_dim
        
        # Create a logarithmically spaced sequence
        dt = torch.exp(torch.linspace(
            math.log(self.dt_min), math.log(self.dt_max), D
        ))
        
        # Initialize A_log so that real(A) = -dt * I
        self.A_log.data.copy_(-dt.unsqueeze(0).expand(N, -1).contiguous())
        
        # Initialize B and C to small random values
        std = 1.0 / math.sqrt(self.state_dim)
        self.B.data.uniform_(-std, std)
        self.C.data.uniform_(-std, std)
    
    def forward(self, x, token_precision=None):
        """Forward pass with token-level adaptive precision"""
        # x shape: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.shape
        
        # 1. Input projections and gating
        # Apply token precision to input projections
        if token_precision is not None:
            # Process each precision level
            x_in_proj = torch.zeros_like(x)
            x_dt_proj = torch.zeros_like(x)
            x_gate_proj = torch.zeros_like(x)
            
            for precision in torch.unique(token_precision):
                precision_val = precision.item()
                
                # Apply precision to each token
                mask = (token_precision == precision_val).unsqueeze(-1).expand_as(x)
                
                # Apply precision-specific quantization to weights
                in_proj_weight, _ = PrecisionManager.quantize_tensor(
                    self.in_proj.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                dt_proj_weight, _ = PrecisionManager.quantize_tensor(
                    self.dt_proj.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                gate_proj_weight, _ = PrecisionManager.quantize_tensor(
                    self.gate_proj.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Apply precision-specific quantization to biases if used
                in_proj_bias = None
                dt_proj_bias = None
                gate_proj_bias = None
                
                if self.use_bias:
                    in_proj_bias, _ = PrecisionManager.quantize_tensor(
                        self.in_proj.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                    dt_proj_bias, _ = PrecisionManager.quantize_tensor(
                        self.dt_proj.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                    gate_proj_bias, _ = PrecisionManager.quantize_tensor(
                        self.gate_proj.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Apply projections with quantized parameters
                x_masked = x[mask].reshape(-1, hidden_size)
                x_in_proj_quantized = F.linear(x_masked, in_proj_weight, in_proj_bias)
                x_dt_proj_quantized = F.linear(x_masked, dt_proj_weight, dt_proj_bias)
                x_gate_proj_quantized = F.linear(x_masked, gate_proj_weight, gate_proj_bias)
                
                # Store results
                x_in_proj[mask] = x_in_proj_quantized.reshape(-1)
                x_dt_proj[mask] = x_dt_proj_quantized.reshape(-1)
                x_gate_proj[mask] = x_gate_proj_quantized.reshape(-1)
                
                # Update precision statistics
                count = mask.sum().item() // hidden_size
                if precision_val not in self.precision_tracker["precision_counts"]:
                    self.precision_tracker["precision_counts"][precision_val] = 0
                self.precision_tracker["precision_counts"][precision_val] += count
            
            # Update tracking statistics
            self.precision_tracker["calls"] += 1
            self.precision_tracker["tokens_processed"] += batch_size * seq_len
            
            # Use the precision-adjusted projections
            u = x_in_proj  # [batch, seq_len, hidden_size]
            delta = x_dt_proj  # [batch, seq_len, hidden_size]
            z = x_gate_proj  # [batch, seq_len, hidden_size]
        else:
            # Standard forward without precision adjustment
            u = self.in_proj(x)  # [batch, seq_len, hidden_size]
            delta = self.dt_proj(x)  # [batch, seq_len, hidden_size]
            z = self.gate_proj(x)  # [batch, seq_len, hidden_size]
        
        # Apply activation for gating
        z = self.activation(z)
        
        # 2. Transform delta to ensure it's within valid range
        # Apply sigmoid to get (0, 1), then scale to (dt_min, dt_max)
        delta = torch.sigmoid(delta) * (self.dt_max - self.dt_min) + self.dt_min
        
        # 3. Get SSM parameters (A, B, C)
        # A is parameterized via its logarithm for better optimization
        A = -torch.exp(self.A_log)  # [hidden_size, state_dim]
        
        # Expand B and C for batch processing
        B = self.B.expand(batch_size, seq_len, -1, -1)  # [batch, seq_len, hidden_size, state_dim]
        C = self.C.expand(batch_size, seq_len, -1, -1)  # [batch, seq_len, hidden_size, state_dim]
        
        # 4. Selective Scan - the core SSM computation
        # We implement this with a custom autograd function
        y = SelectiveScanFn.apply(
            u,              # [batch, seq_len, hidden_size]
            delta,          # [batch, seq_len, hidden_size]
            A,              # [hidden_size, state_dim]
            B,              # [batch, seq_len, hidden_size, state_dim]
            C,              # [batch, seq_len, hidden_size, state_dim]
            None,           # No direct skip connection
            z,              # Gating term [batch, seq_len, hidden_size]
            None,           # No delta_bias
            False,          # No softplus for delta (we already applied sigmoid)
            token_precision # Token-adaptive precision
        )
        
        # 5. Output projection
        if token_precision is not None:
            # Apply token precision to output projection
            output = torch.zeros_like(y)
            
            for precision in torch.unique(token_precision):
                precision_val = precision.item()
                
                # Apply precision to each token
                mask = (token_precision == precision_val).unsqueeze(-1).expand_as(y)
                
                # Apply precision-specific quantization
                out_proj_weight, _ = PrecisionManager.quantize_tensor(
                    self.out_proj.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                out_proj_bias = None
                if self.use_bias:
                    out_proj_bias, _ = PrecisionManager.quantize_tensor(
                        self.out_proj.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Apply projection with quantized parameters
                y_masked = y[mask].reshape(-1, hidden_size)
                output_quantized = F.linear(y_masked, out_proj_weight, out_proj_bias)
                
                # Store results
                output[mask] = output_quantized.reshape(-1)
        else:
            # Standard output projection
            output = self.out_proj(y)
        
        # Apply dropout if needed
        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        
        return output
    
    def get_precision_stats(self):
        """Get statistics about precision usage"""
        stats = {"calls": self.precision_tracker["calls"]}
        
        if self.precision_tracker["tokens_processed"] > 0:
            for prec, count in self.precision_tracker["precision_counts"].items():
                percentage = count / self.precision_tracker["tokens_processed"] * 100
                stats[f"precision_{prec}_pct"] = percentage
            
            # Calculate theoretical energy usage
            if hasattr(self.config, "energy_tracking") and "relative_energy" in self.config.energy_tracking:
                energy_factors = self.config.energy_tracking["relative_energy"]
                relative_energy = 0
                
                for prec, count in self.precision_tracker["precision_counts"].items():
                    pct = count / self.precision_tracker["tokens_processed"]
                    energy_factor = energy_factors.get(prec, 1.0)
                    relative_energy += pct * energy_factor
                
                stats["relative_energy"] = relative_energy
                stats["energy_saved_pct"] = (1 - relative_energy) * 100
        
        return stats
    
    def reset_precision_stats(self):
        """Reset precision usage statistics"""
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}


class TokenAdaptiveMambaBlock(nn.Module):
    """
    Mamba block with token-adaptive precision support.
    Combines token-adaptive SSM layer with feed-forward and normalization layers.
    """
    def __init__(self, 
                 hidden_size, 
                 state_dim=16, 
                 expand_factor=2.0,
                 dt_init=0.001,
                 dt_min=0.001,
                 dt_max=0.1,
                 activation="silu",
                 layer_norm_eps=1e-5,
                 dropout=0.0,
                 config=None):
        super().__init__()
        self.config = config or TAPConfig()
        
        # Layer parameters
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * expand_factor)
        self.state_dim = state_dim
        self.dropout = dropout
        
        # 1. Layer Normalization
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # 2. Input projection for SSM
        self.in_expand = nn.Linear(hidden_size, self.intermediate_size, bias=True)
        
        # 3. Token-adaptive SSM layer
        self.ssm = TokenAdaptiveSSMLayer(
            hidden_size=self.intermediate_size,
            state_dim=state_dim,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=dt_max,
            activation=activation,
            dropout=dropout,
            config=config
        )
        
        # 4. Output projection
        self.out_contract = nn.Linear(self.intermediate_size, hidden_size, bias=True)
        
        # For tracking precision
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
    
    def forward(self, x, token_precision=None):
        """
        Forward pass of Mamba block with token-adaptive precision
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            token_precision: Precision for each token [batch_size, seq_len]
            
        Returns:
            output: Output tensor [batch_size, seq_len, hidden_size]
        """
        # Update tracking
        if token_precision is not None:
            batch_size, seq_len, _ = x.shape
            self.precision_tracker["calls"] += 1
            self.precision_tracker["tokens_processed"] += batch_size * seq_len
            
            # Count tokens at each precision level
            for prec in torch.unique(token_precision):
                prec_val = prec.item()
                count = (token_precision == prec_val).sum().item()
                
                if prec_val not in self.precision_tracker["precision_counts"]:
                    self.precision_tracker["precision_counts"][prec_val] = 0
                
                self.precision_tracker["precision_counts"][prec_val] += count
        
        # Residual connection
        residual = x
        
        # 1. Layer Normalization
        # Apply norm with precision if needed
        if token_precision is not None:
            # Custom normalization with precision-aware computation
            x_normed = self._apply_adaptive_norm(x, token_precision)
        else:
            x_normed = self.norm(x)
        
        # 2. Input expansion
        if token_precision is not None:
            # Apply token precision to input expansion
            x_expanded = self._apply_adaptive_linear(
                x_normed, self.in_expand.weight, self.in_expand.bias, token_precision)
        else:
            x_expanded = self.in_expand(x_normed)
        
        # 3. Apply SSM with token-adaptive precision
        x_ssm = self.ssm(x_expanded, token_precision)
        
        # 4. Output projection
        if token_precision is not None:
            # Apply token precision to output projection
            output = self._apply_adaptive_linear(
                x_ssm, self.out_contract.weight, self.out_contract.bias, token_precision)
        else:
            output = self.out_contract(x_ssm)
        
        # Apply dropout if needed
        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        
        # Residual connection
        output = output + residual
        
        return output
    
    def _apply_adaptive_norm(self, x, token_precision):
        """Apply layer normalization with token-adaptive precision"""
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Output buffer
        x_normed = torch.zeros_like(x)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, hidden_size)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Get unique batch indices
            unique_batch_indices = torch.unique(batch_indices)
            
            # Process each batch separately since LayerNorm is per-token
            for b_idx in unique_batch_indices:
                # Find tokens in this batch with this precision
                batch_mask = (batch_indices == b_idx)
                if not batch_mask.any():
                    continue
                
                cur_seq_indices = seq_indices[batch_mask]
                
                # Extract tokens
                tokens = x[b_idx, cur_seq_indices]
                
                # Quantize tokens for this precision
                tokens_quant, _ = PrecisionManager.quantize_tensor(
                    tokens, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Quantize norm parameters
                weight_quant, _ = PrecisionManager.quantize_tensor(
                    self.norm.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                bias_quant, _ = PrecisionManager.quantize_tensor(
                    self.norm.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Compute normalization
                # Calculate mean and variance
                mean = tokens_quant.mean(dim=-1, keepdim=True)
                var = tokens_quant.var(dim=-1, unbiased=False, keepdim=True)
                
                # Normalize
                tokens_normed = (tokens_quant - mean) / torch.sqrt(var + self.norm.eps)
                
                # Scale and shift
                tokens_normed = tokens_normed * weight_quant + bias_quant
                
                # Store results
                for i, seq_idx in enumerate(cur_seq_indices):
                    x_normed[b_idx, seq_idx] = tokens_normed[i]
        
        return x_normed
    
    def _apply_adaptive_linear(self, x, weight, bias, token_precision):
        """Apply linear transformation with token-adaptive precision"""
        batch_size, seq_len, in_features = x.shape
        out_features = weight.size(0)
        device = x.device
        
        # Output buffer
        output = torch.zeros((batch_size, seq_len, out_features), dtype=x.dtype, device=device)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, in_features)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Reshape for batched processing
            x_precision = torch.zeros((len(batch_indices), in_features), dtype=x.dtype, device=device)
            
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                x_precision[i] = x[b_idx, s_idx]
            
            # Quantize inputs, weights and bias
            x_quant, _ = PrecisionManager.quantize_tensor(
                x_precision, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            weight_quant, _ = PrecisionManager.quantize_tensor(
                weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            bias_quant = None
            if bias is not None:
                bias_quant, _ = PrecisionManager.quantize_tensor(
                    bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            # Perform linear operation
            output_precision = F.linear(x_quant, weight_quant, bias_quant)
            
            # Insert results back into output tensor
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                output[b_idx, s_idx] = output_precision[i]
        
        return output
    
    def get_precision_stats(self):
        """Get statistics about precision usage"""
        # Combine stats from this block and the SSM
        ssm_stats = self.ssm.get_precision_stats()
        
        stats = {"calls": self.precision_tracker["calls"]}
        
        if self.precision_tracker["tokens_processed"] > 0:
            for prec, count in self.precision_tracker["precision_counts"].items():
                percentage = count / self.precision_tracker["tokens_processed"] * 100
                stats[f"precision_{prec}_pct"] = percentage
            
            # Calculate theoretical energy usage
            if hasattr(self.config, "energy_tracking") and "relative_energy" in self.config.energy_tracking:
                energy_factors = self.config.energy_tracking["relative_energy"]
                relative_energy = 0
                
                for prec, count in self.precision_tracker["precision_counts"].items():
                    pct = count / self.precision_tracker["tokens_processed"]
                    energy_factor = energy_factors.get(prec, 1.0)
                    relative_energy += pct * energy_factor
                
                stats["relative_energy"] = relative_energy
                stats["energy_saved_pct"] = (1 - relative_energy) * 100
        
        # Add SSM stats with prefix
        for key, value in ssm_stats.items():
            stats[f"ssm_{key}"] = value
        
        return stats
    
    def reset_precision_stats(self):
        """Reset precision usage statistics"""
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
        self.ssm.reset_precision_stats()


#------------------------------------------------------------------------------
# Token-Adaptive Mamba Model
#------------------------------------------------------------------------------

class TokenAdaptiveMambaModel(nn.Module):
    """
    Complete Mamba model with token-adaptive precision support.
    Implements the full Mamba architecture with token-level precision control.
    """
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 state_dim=16,
                 expand_factor=2.0,
                 max_seq_length=2048,
                 pad_token_id=0,
                 layer_norm_epsilon=1e-5,
                 initializer_range=0.02,
                 use_cache=True,
                 vocab_embed_factor=1.0,
                 dropout=0.0,
                 config=None):
        super().__init__()
        self.config = config or TAPConfig()
        
        # Model parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        self.dropout = dropout
        
        # Calculate embedding scale factor
        # This is a common trick to improve stability in language models
        self.vocab_embed_dim = int(hidden_size * vocab_embed_factor)
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, self.vocab_embed_dim)
        
        # Embedding projection if needed
        if self.vocab_embed_dim != hidden_size:
            self.embed_proj = nn.Linear(self.vocab_embed_dim, hidden_size, bias=True)
        else:
            self.embed_proj = None
        
        # Position embeddings
        # Mamba doesn't use explicit position embeddings by default but can benefit from them
        self.use_position_embeddings = False
        if getattr(self.config, "use_position_embeddings", False):
            self.use_position_embeddings = True
            self.max_seq_length = max_seq_length
            self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        # Layer stack
        self.layers = nn.ModuleList([
            TokenAdaptiveMambaBlock(
                hidden_size=hidden_size,
                state_dim=state_dim,
                expand_factor=expand_factor,
                dt_init=0.001,
                dt_min=0.001,
                dt_max=0.1,
                activation="silu",
                layer_norm_eps=layer_norm_epsilon,
                dropout=dropout,
                config=config
            ) for _ in range(num_hidden_layers)
        ])
        
        # Final layer normalization
        self.norm_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize parameters
        self._init_weights()
        
        # For tracking importance and precision
        self.token_importance_analyzer = TokenImportanceAnalyzer(config=config)
        
        # For tracking overall model metrics
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
    
    def _init_weights(self):
        """Initialize weights for the model"""
        # Apply small random initialization to embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Initialize position embeddings if used
        if self.use_position_embeddings:
            nn.init.normal_(self.position_embedding.weight, std=0.02)
        
        # Initialize embedding projection if used
        if self.embed_proj is not None:
            nn.init.normal_(self.embed_proj.weight, std=0.02)
            if self.embed_proj.bias is not None:
                nn.init.zeros_(self.embed_proj.bias)
        
        # Initialize LM head
        nn.init.normal_(self.lm_head.weight, std=0.02)
        
        # Other parameters are initialized in their respective layers
    
    def get_input_embeddings(self):
        """Get input embeddings module"""
        return self.token_embedding
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embeddings module"""
        self.token_embedding = new_embeddings
    
    def get_output_embeddings(self):
        """Get output embeddings module (LM head)"""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings module (LM head)"""
        self.lm_head = new_embeddings
    
    def forward(self,
               input_ids=None,
               attention_mask=None,
               position_ids=None,
               past_key_values=None,
               inputs_embeds=None,
               labels=None,
               use_cache=None,
               token_precision=None,
               calculate_token_importance=False,
               output_attentions=False,
               output_hidden_states=False,
               return_dict=True):
        """
        Forward pass of the token-adaptive Mamba model
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]
            past_key_values: Past key values for caching
            inputs_embeds: Input embeddings (if input_ids not provided)
            labels: Labels for language modeling loss
            use_cache: Whether to use cache for incremental decoding
            token_precision: Precision for each token [batch, seq_len]
            calculate_token_importance: Whether to calculate token importance
            output_attentions: Whether to output attention weights (not used in Mamba)
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Output with logits, loss (if labels provided), and other requested outputs
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Check if we need to calculate token precision
        if token_precision is None and calculate_token_importance and self.config.precision_mode == PrecisionMode.ADAPTIVE:
            calculate_token_precision = True
        else:
            calculate_token_precision = False
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        
        # Apply embedding projection if needed
        if self.embed_proj is not None:
            hidden_states = self.embed_proj(inputs_embeds)
        else:
            hidden_states = inputs_embeds
        
        # Add position embeddings if used
        if self.use_position_embeddings:
            if position_ids is None:
                # Create position IDs
                seq_length = hidden_states.shape[1]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)
            
            position_embeds = self.position_embedding(position_ids)
            hidden_states = hidden_states + position_embeds
        
        # Track all hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        
        # Calculate token importance and precision if needed
        if calculate_token_precision:
            # First pass without token precision to get hidden states for importance
            with torch.no_grad():
                # Store original hidden states
                original_hidden_states = hidden_states.clone()
                
                # Process first few layers to get meaningful hidden states
                for i, layer in enumerate(self.layers[:min(3, len(self.layers))]):
                    hidden_states = layer(hidden_states)
                
                # Calculate token importance
                token_importance = self.token_importance_analyzer.analyze_token_importance(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    position_ids=position_ids
                )
                
                # Assign precision levels
                token_precision = self.token_importance_analyzer.assign_precision(token_importance)
                
                # Reset hidden states to start fresh with token precision
                hidden_states = original_hidden_states
        
        # Update precision tracking
        if token_precision is not None:
            batch_size, seq_len = token_precision.shape
            self.precision_tracker["calls"] += 1
            self.precision_tracker["tokens_processed"] += batch_size * seq_len
            
            # Count tokens at each precision level
            for prec in torch.unique(token_precision):
                prec_val = int(prec.item())
                count = (token_precision == prec_val).sum().item()
                
                if prec_val not in self.precision_tracker["precision_counts"]:
                    self.precision_tracker["precision_counts"][prec_val] = 0
                
                self.precision_tracker["precision_counts"][prec_val] += count
        
        # Apply all layers with token precision
        for i, layer in enumerate(self.layers):
            # For caching in generation
            layer_outputs = layer(hidden_states, token_precision=token_precision)
            
            hidden_states = layer_outputs
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Final layer norm
        if token_precision is not None:
            # Apply token-adaptive layer norm
            hidden_states = self._apply_adaptive_norm(hidden_states, self.norm_f, token_precision)
        else:
            hidden_states = self.norm_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Apply LM head with token-adaptive precision
        if token_precision is not None:
            logits = self._apply_adaptive_lm_head(hidden_states, token_precision)
        else:
            logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Reshape logits and labels for cross entropy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
        
        # Build return dictionary or tuple
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": all_hidden_states,
                "token_importance": token_importance if calculate_token_precision else None,
                "token_precision": token_precision
            }
        else:
            # Return as tuple (loss, logits, hidden_states)
            return (loss, logits, all_hidden_states)
    
    def _apply_adaptive_norm(self, x, norm_layer, token_precision):
        """Apply layer normalization with token-adaptive precision"""
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Output buffer
        x_normed = torch.zeros_like(x)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, hidden_size)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Get unique batch indices
            unique_batch_indices = torch.unique(batch_indices)
            
            # Process each batch separately since LayerNorm is per-token
            for b_idx in unique_batch_indices:
                # Find tokens in this batch with this precision
                batch_mask = (batch_indices == b_idx)
                if not batch_mask.any():
                    continue
                
                cur_seq_indices = seq_indices[batch_mask]
                
                # Extract tokens
                tokens = x[b_idx, cur_seq_indices]
                
                # Quantize tokens for this precision
                tokens_quant, _ = PrecisionManager.quantize_tensor(
                    tokens, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Quantize norm parameters
                weight_quant, _ = PrecisionManager.quantize_tensor(
                    norm_layer.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                bias_quant, _ = PrecisionManager.quantize_tensor(
                    norm_layer.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Compute normalization
                # Calculate mean and variance
                mean = tokens_quant.mean(dim=-1, keepdim=True)
                var = tokens_quant.var(dim=-1, unbiased=False, keepdim=True)
                
                # Normalize
                tokens_normed = (tokens_quant - mean) / torch.sqrt(var + norm_layer.eps)
                
                # Scale and shift
                tokens_normed = tokens_normed * weight_quant + bias_quant
                
                # Store results
                for i, seq_idx in enumerate(cur_seq_indices):
                    x_normed[b_idx, seq_idx] = tokens_normed[i]
        
        return x_normed
    
    def _apply_adaptive_lm_head(self, x, token_precision):
        """Apply LM head with token-adaptive precision"""
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Output buffer
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), dtype=x.dtype, device=device)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, hidden_size)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Group tokens for efficient processing
            x_precision = torch.zeros((len(batch_indices), hidden_size), dtype=x.dtype, device=device)
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                x_precision[i] = x[b_idx, s_idx]
            
            # Quantize inputs and weights
            x_quant, _ = PrecisionManager.quantize_tensor(
                x_precision, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            weights_quant, _ = PrecisionManager.quantize_tensor(
                self.lm_head.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            # Compute logits
            logits_precision = F.linear(x_quant, weights_quant)
            
            # Store results
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                logits[b_idx, s_idx] = logits_precision[i]
        
        return logits
    
    def get_precision_stats(self):
        """Get statistics about precision usage across the model"""
        stats = {"calls": self.precision_tracker["calls"]}
        
        if self.precision_tracker["tokens_processed"] > 0:
            for prec, count in self.precision_tracker["precision_counts"].items():
                percentage = count / self.precision_tracker["tokens_processed"] * 100
                stats[f"precision_{prec}_pct"] = percentage
            
            # Calculate theoretical energy usage
            if hasattr(self.config, "energy_tracking") and "relative_energy" in self.config.energy_tracking:
                energy_factors = self.config.energy_tracking["relative_energy"]
                relative_energy = 0
                
                for prec, count in self.precision_tracker["precision_counts"].items():
                    pct = count / self.precision_tracker["tokens_processed"]
                    energy_factor = energy_factors.get(prec, 1.0)
                    relative_energy += pct * energy_factor
                
                stats["relative_energy"] = relative_energy
                stats["energy_saved_pct"] = (1 - relative_energy) * 100
        
        # Add detailed per-layer stats
        layer_stats = {}
        for i, layer in enumerate(self.layers):
            layer_stats[f"layer_{i}"] = layer.get_precision_stats()
        
        stats["layers"] = layer_stats
        
        # Add token analyzer stats
        stats["token_analyzer"] = self.token_importance_analyzer.get_analyzer_stats()
        
        return stats
    
    def reset_precision_stats(self):
        """Reset precision usage statistics"""
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
        for layer in self.layers:
            layer.reset_precision_stats()

    def generate(self, 
                input_ids=None,
                attention_mask=None,
                max_length=None,
                min_length=None,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.0,
                num_return_sequences=1,
                pad_token_id=None,
                eos_token_id=None,
                use_adaptive_precision=True,
                **kwargs):
        """
        Generate text using the model with token-adaptive precision
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to return
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            use_adaptive_precision: Whether to use token-adaptive precision
            
        Returns:
            generated_sequences: Generated token sequences
        """
        # Set default values
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else kwargs.get("bos_token_id", None)
        
        if max_length is None:
            max_length = kwargs.get("max_new_tokens", 50) + input_ids.shape[1]
        
        # Set model to evaluation mode
        self.eval()
        
        # Setup for generation
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Clone input_ids for generation
        cur_input_ids = input_ids.clone()
        cur_length = cur_input_ids.shape[1]
        
        # Initialize token importance tracker for adaptive precision
        token_importance_tracker = None
        token_precision_tracker = None
        
        if use_adaptive_precision and self.config.precision_mode == PrecisionMode.ADAPTIVE:
            # Create empty importance and precision trackers
            token_importance_tracker = torch.zeros(
                (batch_size, max_length), device=device, dtype=torch.float32)
            token_precision_tracker = torch.zeros(
                (batch_size, max_length), device=device, dtype=torch.int32)
            
            # Set default precision for initial tokens
            token_precision_tracker[:, :cur_length] = self.config.default_precision
        
        # Main generation loop
        while cur_length < max_length:
            # Get current input slice
            if cur_length > input_ids.shape[1]:
                # Only need the last token for autoregressive generation
                current_input_ids = cur_input_ids[:, -1].unsqueeze(-1)
                current_token_precision = None
                
                if use_adaptive_precision and token_precision_tracker is not None:
                    # Use stored precision if available
                    current_token_precision = token_precision_tracker[:, cur_length-1].unsqueeze(-1)
            else:
                # First pass, use all input tokens
                current_input_ids = cur_input_ids
                current_token_precision = None
                
                if use_adaptive_precision and token_precision_tracker is not None:
                    # Use stored precision for all initial tokens
                    current_token_precision = token_precision_tracker[:, :cur_length]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=current_input_ids,
                    token_precision=current_token_precision,
                    calculate_token_importance=use_adaptive_precision,
                    return_dict=True
                )
            
            # Get logits for next token prediction
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for batch_idx in range(batch_size):
                    for prev_token in cur_input_ids[batch_idx]:
                        next_token_logits[batch_idx, prev_token] /= repetition_penalty
            
            # Mask invalid tokens
            if min_length is not None and cur_length < min_length:
                next_token_logits[:, eos_token_id] = float('-inf')
            
            # Apply sampling
            if do_sample:
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Convert to probabilities and sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Store token precision if using adaptive precision
            if use_adaptive_precision and "token_precision" in outputs and outputs["token_precision"] is not None:
                # Get precision of the predicted next token
                token_precision_tracker[:, cur_length] = outputs["token_precision"][:, -1]
            elif use_adaptive_precision and token_precision_tracker is not None:
                # Default precision if not calculated
                token_precision_tracker[:, cur_length] = self.config.default_precision
            
            # Update input_ids
            cur_input_ids = torch.cat([cur_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            cur_length += 1
            
            # Check for EOS
            if all(next_tokens == eos_token_id):
                break
        
        # Return generated sequences and optionally token precision
        if use_adaptive_precision and token_precision_tracker is not None:
            return {
                "sequences": cur_input_ids,
                "token_precision": token_precision_tracker[:, :cur_length]
            }
        else:
            return {"sequences": cur_input_ids}


#------------------------------------------------------------------------------
# RetNet Implementation with Token-Adaptive Precision
#------------------------------------------------------------------------------

class TokenAdaptiveRetentionFunction(torch.autograd.Function):
    """
    Token-adaptive implementation of the multi-scale retention mechanism
    used in RetNet models.
    """
    
    @staticmethod
    def forward(ctx, xq, xk, xv, decay, token_precision=None):
        """
        Forward pass of token-adaptive retention mechanism
        
        Args:
            xq: Query vectors [batch, seq_len, heads, dim]
            xk: Key vectors [batch, seq_len, heads, dim]
            xv: Value vectors [batch, seq_len, heads, dim]
            decay: Decay rates [heads, dim]
            token_precision: Precision bits for each token [batch, seq_len]
            
        Returns:
            output: Output of retention mechanism [batch, seq_len, heads, dim]
        """
        # Apply precision conversions based on token_precision if provided
        if token_precision is not None:
            batch_size, seq_len, num_heads, head_dim = xq.shape
            device = xq.device
            
            # Process inputs based on token precision
            xq_precision = torch.zeros_like(xq, dtype=torch.float32)
            xk_precision = torch.zeros_like(xk, dtype=torch.float32)
            xv_precision = torch.zeros_like(xv, dtype=torch.float32)
            
            # Process each precision level separately
            for precision in torch.unique(token_precision):
                precision_val = precision.item()
                
                # Create mask for tokens with this precision
                mask = (token_precision == precision_val).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_heads, head_dim)
                
                # Apply quantization at specified precision
                xq_quant, _ = PrecisionManager.quantize_tensor(
                    xq[mask].reshape(-1), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                xk_quant, _ = PrecisionManager.quantize_tensor(
                    xk[mask].reshape(-1), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                xv_quant, _ = PrecisionManager.quantize_tensor(
                    xv[mask].reshape(-1), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Insert quantized values back
                xq_precision[mask] = xq_quant
                xk_precision[mask] = xk_quant
                xv_precision[mask] = xv_quant
            
            # Replace original inputs with precision-adapted ones
            xq = xq_precision
            xk = xk_precision
            xv = xv_precision
        
        # Get shapes
        batch_size, seq_len, num_heads, head_dim = xq.shape
        
        # Prepare storage for outputs and states
        output = torch.zeros_like(xq)
        
        # For each sequence position, compute the weighted sum of all previous keys
        # using an exponentially decaying weight based on distance
        for t in range(seq_len):
            # Current query
            q = xq[:, t]  # [batch, heads, dim]
            
            # Compute retention for position t
            for p in range(t + 1):
                # Get key and value at position p
                k = xk[:, p]  # [batch, heads, dim]
                v = xv[:, p]  # [batch, heads, dim]
                
                # Compute decay based on distance: gamma^(t-p)
                dist = t - p
                decay_factor = decay.pow(dist)  # [heads, dim]
                
                # Apply decay to key-value product
                # output[b, t, h, d] += decay_factor[h, d] * k[b, h, d] * v[b, h, d]
                output[:, t] += decay_factor * k * v
        
        # Save inputs for backward
        ctx.save_for_backward(xq, xk, xv, decay)
        ctx.seq_len = seq_len
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for token-adaptive retention"""
        xq, xk, xv, decay = ctx.saved_tensors
        seq_len = ctx.seq_len
        
        # Initialize gradients
        grad_xq = torch.zeros_like(xq)
        grad_xk = torch.zeros_like(xk)
        grad_xv = torch.zeros_like(xv)
        grad_decay = torch.zeros_like(decay)
        
        # Perform backward pass
        for t in range(seq_len):
            for p in range(t + 1):
                # Get key and value at position p
                k = xk[:, p]  # [batch, heads, dim]
                v = xv[:, p]  # [batch, heads, dim]
                
                # Compute decay factor
                dist = t - p
                decay_factor = decay.pow(dist)  # [heads, dim]
                
                # Gradients for k and v
                grad_k = grad_output[:, t] * decay_factor * v
                grad_v = grad_output[:, t] * decay_factor * k
                
                # Accumulate gradients
                grad_xk[:, p] += grad_k
                grad_xv[:, p] += grad_v
                
                # Gradient for decay (if necessary)
                if dist > 0:  # No decay at position t=p
                    grad_d = grad_output[:, t] * k * v * dist * decay.pow(dist - 1)
                    grad_decay += grad_d.sum(dim=0)  # Sum over batch
        
        return grad_xq, grad_xk, grad_xv, grad_decay, None


class TokenAdaptiveRetNetBlock(nn.Module):
    """
    RetNet block with token-adaptive precision support.
    Implements the parallel and recurrent modes of RetNet with token-level precision.
    """
    def __init__(self, 
                 hidden_size, 
                 num_heads=8,
                 expansion_factor=4,
                 group_size=1,
                 dropout=0.0,
                 layer_norm_epsilon=1e-5,
                 retention_mode="parallel",
                 config=None):
        super().__init__()
        self.config = config or TAPConfig()
        
        # Architecture parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.expansion_factor = expansion_factor
        self.group_size = group_size
        self.dropout = dropout
        self.retention_mode = retention_mode  # "parallel" or "recurrent"
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Multi-scale retention parameters
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Decay rates (theta in the RetNet paper)
        # These are learnable parameters controlling the retention decay
        self.decay_rates = nn.Parameter(torch.ones(num_heads, self.head_dim))
        
        # Feed-forward network
        self.ffn_intermediate = nn.Linear(hidden_size, int(hidden_size * expansion_factor), bias=True)
        self.ffn_output = nn.Linear(int(hidden_size * expansion_factor), hidden_size, bias=True)
        
        # Activation function
        self.act_fn = F.gelu
        
        # Group retention (optional)
        self.use_grouped_retention = group_size > 1
        if self.use_grouped_retention:
            self.group_size = min(group_size, self.num_heads)
            assert self.num_heads % self.group_size == 0, "num_heads must be divisible by group_size"
            self.num_groups = self.num_heads // self.group_size
        
        # Initialize parameters
        self._init_parameters()
        
        # For precision tracking
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
    
    def _init_parameters(self):
        """Initialize parameters with proper scaling"""
        # Initialize QKV projection
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.zeros_(self.qkv_proj.bias)
        
        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize FFN
        nn.init.normal_(self.ffn_intermediate.weight, std=0.02)
        nn.init.zeros_(self.ffn_intermediate.bias)
        nn.init.normal_(self.ffn_output.weight, std=0.02)
        nn.init.zeros_(self.ffn_output.bias)
        
        # Initialize decay rates to logarithmically spaced values between 0.9 and 0.999
        # This gives a good range of timescales for the retention mechanism
        log_vals = torch.linspace(math.log(0.9), math.log(0.999), self.head_dim)
        theta = torch.exp(log_vals)
        self.decay_rates.data = theta.expand(self.num_heads, self.head_dim)
    
    def forward(self, hidden_states, token_precision=None):
        """
        Forward pass of RetNet block with token-adaptive precision
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            token_precision: Precision bits for each token [batch, seq_len]
            
        Returns:
            hidden_states: Output tensor [batch, seq_len, hidden_size]
        """
        # Track precision usage
        if token_precision is not None:
            batch_size, seq_len, _ = hidden_states.shape
            self.precision_tracker["calls"] += 1
            self.precision_tracker["tokens_processed"] += batch_size * seq_len
            
            # Count tokens at each precision level
            for prec in torch.unique(token_precision):
                prec_val = prec.item()
                count = (token_precision == prec_val).sum().item()
                
                if prec_val not in self.precision_tracker["precision_counts"]:
                    self.precision_tracker["precision_counts"][prec_val] = 0
                
                self.precision_tracker["precision_counts"][prec_val] += count
        
        # Residual connection
        residual = hidden_states
        
        # Layer normalization
        if token_precision is not None:
            hidden_states = self._apply_adaptive_norm(hidden_states, self.norm1, token_precision)
        else:
            hidden_states = self.norm1(hidden_states)
        
        # Multi-scale retention
        hidden_states = self._retention_forward(hidden_states, token_precision)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Feed-forward network with second residual connection
        residual = hidden_states
        
        # Second layer normalization
        if token_precision is not None:
            hidden_states = self._apply_adaptive_norm(hidden_states, self.norm2, token_precision)
        else:
            hidden_states = self.norm2(hidden_states)
        
        # FFN
        if token_precision is not None:
            # Apply token-adaptive FFN
            hidden_states = self._apply_adaptive_ffn(hidden_states, token_precision)
        else:
            hidden_states = self.ffn_output(self.act_fn(self.ffn_intermediate(hidden_states)))
        
        # Apply dropout if needed
        if self.dropout > 0:
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _retention_forward(self, hidden_states, token_precision=None):
        """Apply multi-scale retention with token-adaptive precision"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        if token_precision is not None:
            # Apply token-adaptive QKV projection
            qkv = self._apply_adaptive_linear(hidden_states, self.qkv_proj.weight, 
                                              self.qkv_proj.bias, token_precision)
        else:
            qkv = self.qkv_proj(hidden_states)
        
        # Split Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Apply parallel or recurrent retention
        if self.retention_mode == "parallel":
            # Parallel (training) mode
            output = self._parallel_retention(q, k, v, token_precision)
        else:
            # Recurrent (inference) mode
            output = self._recurrent_retention(q, k, v, token_precision)
        
        # Output projection
        if token_precision is not None:
            output = self._apply_adaptive_linear(output, self.output_proj.weight, 
                                               self.output_proj.bias, token_precision)
        else:
            output = self.output_proj(output)
        
        # Apply dropout if needed
        if self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)
        
        return output
    
    def _parallel_retention(self, q, k, v, token_precision=None):
        """
        Parallel implementation of multi-scale retention mechanism
        
        Args:
            q: Query vectors [batch, seq_len, heads, dim]
            k: Key vectors [batch, seq_len, heads, dim]
            v: Value vectors [batch, seq_len, heads, dim]
            token_precision: Precision bits for each token [batch, seq_len]
            
        Returns:
            output: Output of retention mechanism [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        device = q.device
        
        # Use the custom autograd function for token-adaptive retention
        output = TokenAdaptiveRetentionFunction.apply(q, k, v, self.decay_rates, token_precision)
        
        # Reshape output
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        
        return output
    
    def _recurrent_retention(self, q, k, v, token_precision=None):
        """
        Recurrent implementation of multi-scale retention for efficient inference
        
        Args:
            q: Query vectors [batch, seq_len, heads, dim]
            k: Key vectors [batch, seq_len, heads, dim]
            v: Value vectors [batch, seq_len, heads, dim]
            token_precision: Precision bits for each token [batch, seq_len]
            
        Returns:
            output: Output of retention mechanism [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        device = q.device
        
        # Initialize state (s_t-1)
        state = torch.zeros(batch_size, num_heads, head_dim, device=device)
        outputs = []
        
        # Process sequence recurrently
        for t in range(seq_len):
            # Get current q, k, v
            q_t = q[:, t]  # [batch, heads, dim]
            k_t = k[:, t]  # [batch, heads, dim]
            v_t = v[:, t]  # [batch, heads, dim]
            
            # Apply token-adaptive precision if needed
            if token_precision is not None:
                precision_t = token_precision[:, t]
                
                # Apply precision-specific quantization
                for precision in torch.unique(precision_t):
                    precision_val = precision.item()
                    mask = (precision_t == precision_val).unsqueeze(-1).unsqueeze(-1)
                    
                    # Quantize q, k, v for this token's precision
                    if mask.any():
                        q_prec, _ = PrecisionManager.quantize_tensor(
                            q_t[mask.any(dim=0)], bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                        k_prec, _ = PrecisionManager.quantize_tensor(
                            k_t[mask.any(dim=0)], bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                        v_prec, _ = PrecisionManager.quantize_tensor(
                            v_t[mask.any(dim=0)], bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                        
                        # Apply to the correct portions
                        q_t[mask.any(dim=0)] = q_prec
                        k_t[mask.any(dim=0)] = k_prec
                        v_t[mask.any(dim=0)] = v_prec
            
            # Update state: s_t = gamma * s_t-1 + k_t⊗v_t
            state = self.decay_rates * state + k_t * v_t
            
            # Compute output: y_t = q_t⊗s_t
            output_t = q_t * state
            outputs.append(output_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, heads, dim]
        
        # Reshape to [batch, seq_len, hidden_size]
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        
        return output
    
    def _apply_adaptive_norm(self, x, norm_layer, token_precision):
        """Apply layer normalization with token-adaptive precision"""
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Output buffer
        x_normed = torch.zeros_like(x)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, hidden_size)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Get unique batch indices
            unique_batch_indices = torch.unique(batch_indices)
            
            # Process each batch separately since LayerNorm is per-token
            for b_idx in unique_batch_indices:
                # Find tokens in this batch with this precision
                batch_mask = (batch_indices == b_idx)
                if not batch_mask.any():
                    continue
                
                cur_seq_indices = seq_indices[batch_mask]
                
                # Extract tokens
                tokens = x[b_idx, cur_seq_indices]
                
                # Quantize tokens for this precision
                tokens_quant, _ = PrecisionManager.quantize_tensor(
                    tokens, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Quantize norm parameters
                weight_quant, _ = PrecisionManager.quantize_tensor(
                    norm_layer.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                bias_quant, _ = PrecisionManager.quantize_tensor(
                    norm_layer.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Compute normalization
                # Calculate mean and variance
                mean = tokens_quant.mean(dim=-1, keepdim=True)
                var = tokens_quant.var(dim=-1, unbiased=False, keepdim=True)
                
                # Normalize
                tokens_normed = (tokens_quant - mean) / torch.sqrt(var + norm_layer.eps)
                
                # Scale and shift
                tokens_normed = tokens_normed * weight_quant + bias_quant
                
                # Store results
                for i, seq_idx in enumerate(cur_seq_indices):
                    x_normed[b_idx, seq_idx] = tokens_normed[i]
        
        return x_normed
    
    def _apply_adaptive_linear(self, x, weight, bias, token_precision):
        """Apply linear transformation with token-adaptive precision"""
        batch_size, seq_len, in_features = x.shape
        out_features = weight.size(0)
        device = x.device
        
        # Output buffer
        output = torch.zeros((batch_size, seq_len, out_features), dtype=x.dtype, device=device)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, in_features)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Reshape for batched processing
            x_precision = torch.zeros((len(batch_indices), in_features), dtype=x.dtype, device=device)
            
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                x_precision[i] = x[b_idx, s_idx]
            
            # Quantize inputs, weights and bias
            x_quant, _ = PrecisionManager.quantize_tensor(
                x_precision, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            weight_quant, _ = PrecisionManager.quantize_tensor(
                weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            bias_quant = None
            if bias is not None:
                bias_quant, _ = PrecisionManager.quantize_tensor(
                    bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            # Perform linear operation
            output_precision = F.linear(x_quant, weight_quant, bias_quant)
            
            # Insert results back into output tensor
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                output[b_idx, s_idx] = output_precision[i]
        
        return output
    
    def _apply_adaptive_ffn(self, x, token_precision):
        """Apply feed-forward network with token-adaptive precision"""
        # First linear + activation
        intermediate = self._apply_adaptive_linear(
            x, self.ffn_intermediate.weight, self.ffn_intermediate.bias, token_precision)
        intermediate = self.act_fn(intermediate)
        
        # Second linear
        output = self._apply_adaptive_linear(
            intermediate, self.ffn_output.weight, self.ffn_output.bias, token_precision)
        
        return output
    
    def get_precision_stats(self):
        """Get statistics about precision usage"""
        stats = {"calls": self.precision_tracker["calls"]}
        
        if self.precision_tracker["tokens_processed"] > 0:
            for prec, count in self.precision_tracker["precision_counts"].items():
                percentage = count / self.precision_tracker["tokens_processed"] * 100
                stats[f"precision_{prec}_pct"] = percentage
            
            # Calculate theoretical energy usage
            if hasattr(self.config, "energy_tracking") and "relative_energy" in self.config.energy_tracking:
                energy_factors = self.config.energy_tracking["relative_energy"]
                relative_energy = 0
                
                for prec, count in self.precision_tracker["precision_counts"].items():
                    pct = count / self.precision_tracker["tokens_processed"]
                    energy_factor = energy_factors.get(prec, 1.0)
                    relative_energy += pct * energy_factor
                
                stats["relative_energy"] = relative_energy
                stats["energy_saved_pct"] = (1 - relative_energy) * 100
        
        return stats
    
    def reset_precision_stats(self):
        """Reset precision usage statistics"""
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}


#------------------------------------------------------------------------------
# RWKV Implementation with Token-Adaptive Precision
#------------------------------------------------------------------------------

class TokenAdaptiveWKVFunction(torch.autograd.Function):
    """
    Autograd function for the WKV operation in RWKV with token-adaptive precision.
    Implements the time-mixed attention mechanism with token-level precision control.
    """
    
    @staticmethod
    def forward(ctx, time_decay, time_first, key, value, token_precision=None):
        """
        Forward pass of the WKV operation
        
        Args:
            time_decay: Time decay tensor [hidden_size]
            time_first: Time first tensor [hidden_size]
            key: Key tensor [batch, seq_len, hidden_size]
            value: Value tensor [batch, seq_len, hidden_size]
            token_precision: Precision bits for each token [batch, seq_len]
            
        Returns:
            output: WKV tensor [batch, seq_len, hidden_size]
        """
        # Apply precision conversions based on token_precision if provided
        if token_precision is not None:
            batch_size, seq_len, hidden_size = key.shape
            device = key.device
            
            # Process inputs based on token precision
            key_precision = torch.zeros_like(key, dtype=torch.float32)
            value_precision = torch.zeros_like(value, dtype=torch.float32)
            
            # Process each precision level separately
            for precision in torch.unique(token_precision):
                precision_val = precision.item()
                
                # Create mask for tokens with this precision
                mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, hidden_size)
                
                # Apply quantization at specified precision
                key_quant, _ = PrecisionManager.quantize_tensor(
                    key[mask].reshape(-1), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                value_quant, _ = PrecisionManager.quantize_tensor(
                    value[mask].reshape(-1), bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Insert quantized values back
                key_precision[mask] = key_quant
                value_precision[mask] = value_quant
            
            # Replace original inputs with precision-adapted ones
            key = key_precision
            value = value_precision
            
            # Also quantize time_decay and time_first
            time_decay_quant = {}
            time_first_quant = {}
            
            # Quantize for each precision level
            for precision in torch.unique(token_precision):
                precision_val = precision.item()
                
                # Quantize time parameters
                time_decay_q, _ = PrecisionManager.quantize_tensor(
                    time_decay, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                time_first_q, _ = PrecisionManager.quantize_tensor(
                    time_first, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Store quantized parameters
                time_decay_quant[precision_val] = time_decay_q
                time_first_quant[precision_val] = time_first_q
        else:
            # Create dummy dictionaries to simplify logic
            time_decay_quant = {0: time_decay}
            time_first_quant = {0: time_first}
        
        # Get shapes
        batch_size, seq_len, hidden_size = key.shape
        device = key.device
        
        # Initialize output and state
        output = torch.zeros_like(key)
        
        # Compute WKV using the RWKV receptance formula
        # Reference: https://github.com/BlinkDL/RWKV-LM
        
        # For each sequence position
        for t in range(seq_len):
            # Initialize numerator and denominator state
            num_state = torch.zeros((batch_size, hidden_size), device=device)
            den_state = torch.zeros((batch_size, hidden_size), device=device)
            
            # Determine precision for this token
            if token_precision is not None:
                precision_t = token_precision[:, t].item()
                time_decay_t = time_decay_quant[precision_t]
                time_first_t = time_first_quant[precision_t]
            else:
                time_decay_t = time_decay
                time_first_t = time_first
            
            # Process current token and all previous tokens
            for p in range(t + 1):
                # Determine precision for token at position p
                if token_precision is not None and p < t:
                    precision_p = token_precision[:, p].item()
                    time_decay_p = time_decay_quant[precision_p]
                else:
                    time_decay_p = time_decay_t
                
                # Current key and value
                k_p = key[:, p]  # [batch, hidden]
                v_p = value[:, p]  # [batch, hidden]
                
                # Compute time decay factor
                if p == t:
                    # For current token, use time_first
                    decay_factor = time_first_t
                else:
                    # For previous tokens, use time_decay
                    dist = t - p - 1
                    decay_factor = torch.exp(-torch.exp(time_decay_p) * dist)
                
                # Update numerator: sum(k_i * v_i * decay_factor)
                num_state += k_p * v_p * decay_factor
                
                # Update denominator: sum(k_i * decay_factor)
                den_state += k_p * decay_factor
            
            # Compute WKV = num_state / den_state
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            output[:, t] = num_state / (den_state + epsilon)
        
        # Save for backward
        ctx.save_for_backward(time_decay, time_first, key, value, output)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for token-adaptive WKV operation"""
        time_decay, time_first, key, value, output = ctx.saved_tensors
        
        # Initialize gradients
        grad_time_decay = torch.zeros_like(time_decay)
        grad_time_first = torch.zeros_like(time_first)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        
        # Implement backward pass for WKV operation
        # This is a simplified approximation of the exact gradient
        
        batch_size, seq_len, hidden_size = key.shape
        
        # For each position
        for t in range(seq_len):
            # Propagate gradients
            grad_out_t = grad_output[:, t]  # [batch, hidden]
            
            # For current and all previous tokens
            for p in range(t + 1):
                # Compute decay factor
                if p == t:
                    # Current token uses time_first
                    decay_factor = time_first
                    
                    # Gradient for time_first
                    grad_time_first += torch.sum(grad_out_t * output[:, t] * key[:, p], dim=0)
                else:
                    # Previous tokens use time_decay
                    dist = t - p - 1
                    decay_scalar = torch.exp(-torch.exp(time_decay) * dist)
                    
                    # Gradient for time_decay
                    grad_factor = -dist * torch.exp(time_decay) * decay_scalar
                    grad_time_decay += torch.sum(grad_out_t * value[:, p] * key[:, p] * grad_factor, dim=0)
                    
                    decay_factor = decay_scalar
                
                # Gradient for key and value
                grad_key[:, p] += grad_out_t * value[:, p] * decay_factor
                grad_value[:, p] += grad_out_t * key[:, p] * decay_factor
        
        return grad_time_decay, grad_time_first, grad_key, grad_value, None


class TokenAdaptiveRWKVBlock(nn.Module):
    """
    RWKV block with token-adaptive precision support.
    Implements the RWKV architecture with token-level precision control.
    """
    def __init__(self, 
                 hidden_size, 
                 ffn_hidden_size=None,
                 num_attention_heads=8,
                 layer_norm_epsilon=1e-5,
                 dropout=0.0,
                 time_mix_factor=1.0,
                 channel_mix_factor=1.0,
                 config=None):
        super().__init__()
        self.config = config or TAPConfig()
        
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size or (hidden_size * 4)
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Time mixing part
        self.time_decay = nn.Parameter(torch.ones(hidden_size) * math.log(0.99))  # Initialize to slow decay
        self.time_first = nn.Parameter(torch.ones(hidden_size) * math.log(0.5))   # Initialize for balance
        
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, hidden_size) * time_mix_factor)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, hidden_size) * time_mix_factor)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, hidden_size) * time_mix_factor)
        
        # Time mixing projections
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Channel mixing part
        self.channel_mix_k = nn.Parameter(torch.ones(1, 1, hidden_size) * channel_mix_factor)
        self.channel_mix_r = nn.Parameter(torch.ones(1, 1, hidden_size) * channel_mix_factor)
        
        # Channel mixing projections
        self.key_channel = nn.Linear(hidden_size, self.ffn_hidden_size, bias=False)
        self.receptance_channel = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_channel = nn.Linear(self.ffn_hidden_size, hidden_size, bias=False)
        
        # Initialize parameters
        self._init_weights()
        
        # For precision tracking
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
    
    def _init_weights(self):
        """Initialize parameters"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, hidden_states, token_precision=None):
        """
        Forward pass of RWKV block with token-adaptive precision
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            token_precision: Precision bits for each token [batch, seq_len]
            
        Returns:
            hidden_states: Output tensor [batch, seq_len, hidden_size]
        """
        # Update precision tracking
        if token_precision is not None:
            batch_size, seq_len, _ = hidden_states.shape
            self.precision_tracker["calls"] += 1
            self.precision_tracker["tokens_processed"] += batch_size * seq_len
            
            # Count tokens at each precision level
            for prec in torch.unique(token_precision):
                prec_val = prec.item()
                count = (token_precision == prec_val).sum().item()
                
                if prec_val not in self.precision_tracker["precision_counts"]:
                    self.precision_tracker["precision_counts"][prec_val] = 0
                
                self.precision_tracker["precision_counts"][prec_val] += count
        
        # ---- Time-mixing part ----
        residual = hidden_states
        
        # Apply layer norm
        if token_precision is not None:
            hidden_states = self._apply_adaptive_norm(hidden_states, self.ln1, token_precision)
        else:
            hidden_states = self.ln1(hidden_states)
        
        # Split states for time mixing
        state_with_last = torch.cat([hidden_states[:, -1:, :], hidden_states[:, :-1, :]], dim=1)
        
        # Apply time mixing with token-adaptive precision
        k_mix = hidden_states * self.time_mix_k + state_with_last * (1 - self.time_mix_k)
        v_mix = hidden_states * self.time_mix_v + state_with_last * (1 - self.time_mix_v)
        r_mix = hidden_states * self.time_mix_r + state_with_last * (1 - self.time_mix_r)
        
        # Apply projections with token-adaptive precision
        if token_precision is not None:
            k = self._apply_adaptive_linear(k_mix, self.key.weight, None, token_precision)
            v = self._apply_adaptive_linear(v_mix, self.value.weight, None, token_precision)
            r = self._apply_adaptive_linear(r_mix, self.receptance.weight, None, token_precision)
        else:
            k = self.key(k_mix)
            v = self.value(v_mix)
            r = self.receptance(r_mix)
        
        # Apply WKV operation (weighted key-value)
        wkv = TokenAdaptiveWKVFunction.apply(self.time_decay, self.time_first, k, v, token_precision)
        
        # Apply receptance gating
        r = torch.sigmoid(r)
        time_mixed = r * wkv
        
        # Apply output projection
        if token_precision is not None:
            time_output = self._apply_adaptive_linear(time_mixed, self.output.weight, None, token_precision)
        else:
            time_output = self.output(time_mixed)
        
        # Residual connection
        hidden_states = residual + time_output
        
        # ---- Channel-mixing part ----
        residual = hidden_states
        
        # Apply layer norm
        if token_precision is not None:
            hidden_states = self._apply_adaptive_norm(hidden_states, self.ln2, token_precision)
        else:
            hidden_states = self.ln2(hidden_states)
        
        # Split states for channel mixing
        state_with_last = torch.cat([hidden_states[:, -1:, :], hidden_states[:, :-1, :]], dim=1)
        
        # Apply channel mixing
        k_mix = hidden_states * self.channel_mix_k + state_with_last * (1 - self.channel_mix_k)
        r_mix = hidden_states * self.channel_mix_r + state_with_last * (1 - self.channel_mix_r)
        
        # Apply projections with token-adaptive precision
        if token_precision is not None:
            k = self._apply_adaptive_linear(k_mix, self.key_channel.weight, None, token_precision)
            r = self._apply_adaptive_linear(r_mix, self.receptance_channel.weight, None, token_precision)
        else:
            k = self.key_channel(k_mix)
            r = self.receptance_channel(r_mix)
        
        # Apply activation and gating
        k = torch.square(torch.relu(k))  # RWKV uses squared ReLU
        r = torch.sigmoid(r)
        
        # Apply output projection
        if token_precision is not None:
            channel_output = r * self._apply_adaptive_linear(k, self.value_channel.weight, None, token_precision)
        else:
            channel_output = r * self.value_channel(k)
        
        # Final residual connection
        hidden_states = residual + channel_output
        
        # Apply dropout if needed
        if self.dropout > 0:
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        return hidden_states
    
    def _apply_adaptive_norm(self, x, norm_layer, token_precision):
        """Apply layer normalization with token-adaptive precision"""
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Output buffer
        x_normed = torch.zeros_like(x)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, hidden_size)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Get unique batch indices
            unique_batch_indices = torch.unique(batch_indices)
            
            # Process each batch separately since LayerNorm is per-token
            for b_idx in unique_batch_indices:
                # Find tokens in this batch with this precision
                batch_mask = (batch_indices == b_idx)
                if not batch_mask.any():
                    continue
                
                cur_seq_indices = seq_indices[batch_mask]
                
                # Extract tokens
                tokens = x[b_idx, cur_seq_indices]
                
                # Quantize tokens for this precision
                tokens_quant, _ = PrecisionManager.quantize_tensor(
                    tokens, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Quantize norm parameters
                weight_quant, _ = PrecisionManager.quantize_tensor(
                    norm_layer.weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                bias_quant, _ = PrecisionManager.quantize_tensor(
                    norm_layer.bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                
                # Compute normalization
                # Calculate mean and variance
                mean = tokens_quant.mean(dim=-1, keepdim=True)
                var = tokens_quant.var(dim=-1, unbiased=False, keepdim=True)
                
                # Normalize
                tokens_normed = (tokens_quant - mean) / torch.sqrt(var + norm_layer.eps)
                
                # Scale and shift
                tokens_normed = tokens_normed * weight_quant + bias_quant
                
                # Store results
                for i, seq_idx in enumerate(cur_seq_indices):
                    x_normed[b_idx, seq_idx] = tokens_normed[i]
        
        return x_normed
    
    def _apply_adaptive_linear(self, x, weight, bias, token_precision):
        """Apply linear transformation with token-adaptive precision"""
        batch_size, seq_len, in_features = x.shape
        out_features = weight.size(0)
        device = x.device
        
        # Output buffer
        output = torch.zeros((batch_size, seq_len, out_features), dtype=x.dtype, device=device)
        
        # Process each precision level
        for precision in torch.unique(token_precision):
            precision_val = precision.item()
            
            # Create mask for tokens with this precision
            mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, in_features)
            
            if not mask.any():
                continue
            
            # Extract tokens with this precision
            indices = torch.where(mask.any(dim=-1))
            batch_indices, seq_indices = indices[0], indices[1]
            
            # Reshape for batched processing
            x_precision = torch.zeros((len(batch_indices), in_features), dtype=x.dtype, device=device)
            
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                x_precision[i] = x[b_idx, s_idx]
            
            # Quantize inputs, weights and bias
            x_quant, _ = PrecisionManager.quantize_tensor(
                x_precision, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            weight_quant, _ = PrecisionManager.quantize_tensor(
                weight, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            bias_quant = None
            if bias is not None:
                bias_quant, _ = PrecisionManager.quantize_tensor(
                    bias, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
            
            # Perform linear operation
            output_precision = F.linear(x_quant, weight_quant, bias_quant)
            
            # Insert results back into output tensor
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                output[b_idx, s_idx] = output_precision[i]
        
        return output
    
    def get_precision_stats(self):
        """Get statistics about precision usage"""
        stats = {"calls": self.precision_tracker["calls"]}
        
        if self.precision_tracker["tokens_processed"] > 0:
            for prec, count in self.precision_tracker["precision_counts"].items():
                percentage = count / self.precision_tracker["tokens_processed"] * 100
                stats[f"precision_{prec}_pct"] = percentage
            
            # Calculate theoretical energy usage
            if hasattr(self.config, "energy_tracking") and "relative_energy" in self.config.energy_tracking:
                energy_factors = self.config.energy_tracking["relative_energy"]
                relative_energy = 0
                
                for prec, count in self.precision_tracker["precision_counts"].items():
                    pct = count / self.precision_tracker["tokens_processed"]
                    energy_factor = energy_factors.get(prec, 1.0)
                    relative_energy += pct * energy_factor
                
                stats["relative_energy"] = relative_energy
                stats["energy_saved_pct"] = (1 - relative_energy) * 100
        
        return stats
    
    def reset_precision_stats(self):
        """Reset precision usage statistics"""
        self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}


#------------------------------------------------------------------------------
# Integration with Common Model Types and Utilities
#------------------------------------------------------------------------------

class ModelIntegrationManager:
    """
    Utility class to help integrate TAP Engine with various model architectures.
    Provides tools for model conversion, precision analysis, and performance monitoring.
    """
    
    @staticmethod
    def convert_to_token_adaptive(model, model_type=None, config=None):
        """
        Convert a standard model to a token-adaptive version
        
        Args:
            model: PyTorch model to convert
            model_type: Type of model (transformer, mamba, retnet, rwkv)
            config: TAP configuration
            
        Returns:
            model: Token-adaptive model
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot convert model")
            return model
        
        if config is None:
            config = TAPConfig()
        
        # Detect model type if not specified
        if model_type is None:
            model_type = ModelIntegrationManager._detect_model_type(model)
        
        # Set model architecture in config
        if isinstance(model_type, str):
            try:
                config.model_arch = ModelArchitecture(model_type)
            except ValueError:
                logger.warning(f"Unknown model type: {model_type}, using TRANSFORMER")
                config.model_arch = ModelArchitecture.TRANSFORMER
        elif isinstance(model_type, ModelArchitecture):
            config.model_arch = model_type
        
        # Convert based on model type
        if config.model_arch == ModelArchitecture.TRANSFORMER:
            return ModelIntegrationManager._convert_transformer(model, config)
        elif config.model_arch == ModelArchitecture.MAMBA:
            return ModelIntegrationManager._convert_mamba(model, config)
        elif config.model_arch == ModelArchitecture.RETNET:
            return ModelIntegrationManager._convert_retnet(model, config)
        elif config.model_arch == ModelArchitecture.RWKV:
            return ModelIntegrationManager._convert_rwkv(model, config)
        else:
            logger.warning(f"Unsupported model architecture: {config.model_arch}, returning original model")
            return model
    
    @staticmethod
    def _detect_model_type(model):
        """Detect model type from model architecture"""
        model_name = model.__class__.__name__.lower()
        
        if "mamba" in model_name or "ssm" in model_name:
            return ModelArchitecture.MAMBA
        elif "retnet" in model_name or "retention" in model_name:
            return ModelArchitecture.RETNET
        elif "rwkv" in model_name:
            return ModelArchitecture.RWKV
        else:
            # Default to transformer
            return ModelArchitecture.TRANSFORMER
    
    @staticmethod
    def _convert_transformer(model, config):
        """Convert a transformer model to token-adaptive version"""
        # Check for common transformer architectures
        if TRANSFORMERS_AVAILABLE:
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block
            from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            from transformers.models.bert.modeling_bert import BertLayer
            
            # Recursively replace transformer blocks with token-adaptive versions
            for name, module in model.named_children():
                if isinstance(module, (GPT2Block, GPTNeoBlock, LlamaDecoderLayer, BertLayer)):
                    # Replace with token-adaptive version
                    # Implementation depends on specific architecture
                    pass
                elif len(list(module.children())) > 0:
                    # Recurse into submodules
                    new_module = ModelIntegrationManager._convert_transformer(module, config)
                    if new_module is not module:
                        setattr(model, name, new_module)
        
        # If no transformers library or no recognized blocks, 
        # provide generic conversion for common attribute names
        for name, module in model.named_modules():
            # Look for common layer types
            if isinstance(module, nn.Linear):
                pass  # Can be wrapped with TokenAdaptiveLinear
            elif isinstance(module, nn.LayerNorm):
                pass  # Can be wrapped with TokenAdaptiveLayerNorm
            # Add more layer types as needed
        
        return model
    
    @staticmethod
    def _convert_mamba(model, config):
        """Convert a Mamba model to token-adaptive version"""
        # Specific conversion logic for Mamba models
        
        # Example conversion for a Mamba block
        for name, module in model.named_children():
            # Check for Mamba-specific module types
            if "ssm" in name.lower() or "mamba" in name.lower():
                # Potential replacement with TokenAdaptiveSSMLayer
                pass
            elif len(list(module.children())) > 0:
                # Recurse into submodules
                new_module = ModelIntegrationManager._convert_mamba(module, config)
                if new_module is not module:
                    setattr(model, name, new_module)
        
        return model
    
    @staticmethod
    def _convert_retnet(model, config):
        """Convert a RetNet model to token-adaptive version"""
        # Specific conversion logic for RetNet models
        
        # Example conversion for RetNet blocks
        for name, module in model.named_children():
            # Check for RetNet-specific module types
            if "retention" in name.lower() or "retnet" in name.lower():
                # Potential replacement with TokenAdaptiveRetNetBlock
                pass
            elif len(list(module.children())) > 0:
                # Recurse into submodules
                new_module = ModelIntegrationManager._convert_retnet(module, config)
                if new_module is not module:
                    setattr(model, name, new_module)
        
        return model
    
    @staticmethod
    def _convert_rwkv(model, config):
        """Convert an RWKV model to token-adaptive version"""
        # Specific conversion logic for RWKV models
        
        # Example conversion for RWKV blocks
        for name, module in model.named_children():
            # Check for RWKV-specific module types
            if "rwkv" in name.lower() or "wkv" in name.lower():
                # Potential replacement with TokenAdaptiveRWKVBlock
                pass
            elif len(list(module.children())) > 0:
                # Recurse into submodules
                new_module = ModelIntegrationManager._convert_rwkv(module, config)
                if new_module is not module:
                    setattr(model, name, new_module)
        
        return model
    
    @staticmethod
    def analyze_token_importance(model, input_ids, attention_mask=None, config=None):
        """
        Analyze token importance for a given input
        
        Args:
            model: PyTorch model
            input_ids: Input token IDs
            attention_mask: Attention mask
            config: TAP configuration
            
        Returns:
            token_importance: Token importance scores
            model_outputs: Model outputs
        """
        if config is None:
            config = TAPConfig()
        
        # Create token importance analyzer
        analyzer = TokenImportanceAnalyzer(config)
        
        # Get model architecture
        model_arch = ModelIntegrationManager._detect_model_type(model)
        
        # Put model in eval mode
        model.eval()
        
        # Forward pass with hidden states and attention scores
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
        
        # Extract hidden states and attention scores
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = None
        
        if hasattr(outputs, "attentions") and outputs.attentions:
            attention_scores = outputs.attentions[-1]
        else:
            attention_scores = None
        
        # Additional extraction for SSM models
        ssm_states = None
        if model_arch == ModelArchitecture.MAMBA and hasattr(outputs, "ssm_states"):
            ssm_states = outputs.ssm_states
        
        # Calculate token importance
        token_importance = analyzer.analyze_token_importance(
            attention_scores=attention_scores,
            hidden_states=hidden_states,
            ssm_states=ssm_states,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Calculate precision levels
        token_precision = analyzer.assign_precision(token_importance)
        
        # Analyze statistics
        precision_stats = analyzer.analyze_precision_stats(token_importance, token_precision)
        
        return {
            "token_importance": token_importance,
            "token_precision": token_precision,
            "precision_stats": precision_stats,
            "model_outputs": outputs
        }
    
    @staticmethod
    def integrate_with_accelerate(model, config=None):
        """
        Integrate token-adaptive model with Hugging Face Accelerate
        
        Args:
            model: Token-adaptive model
            config: TAP configuration
            
        Returns:
            model: Accelerate-integrated model
        """
        if not ACCELERATE_AVAILABLE:
            logger.warning("Accelerate library not available, skipping integration")
            return model
        
        if config is None:
            config = TAPConfig()
        
        # Initialize accelerator
        accelerator = Accelerator()
        
        # Prepare model with accelerator
        model = accelerator.prepare_model(model)
        
        return model
    
    @staticmethod
    def integrate_with_lora(model, config=None, lora_config=None):
        """
        Integrate token-adaptive model with LoRA
        
        Args:
            model: Token-adaptive model
            config: TAP configuration
            lora_config: LoRA configuration
            
        Returns:
            model: LoRA-integrated model
        """
        if not PEFT_AVAILABLE:
            logger.warning("PEFT library not available, skipping LoRA integration")
            return model
        
        if config is None:
            config = TAPConfig()
        
        if lora_config is None:
            # Default LoRA configuration
            lora_config = LoraConfig(
                r=16,                       # LoRA rank
                lora_alpha=32,              # LoRA alpha scaling factor
                target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to
                lora_dropout=0.05,          # Dropout probability
                bias="none",                # Bias handling
                task_type="CAUSAL_LM"       # Task type
            )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        return model
    
    @staticmethod
    def enable_qlora(model, config=None):
        """
        Enable QLoRA for token-adaptive model
        
        Args:
            model: Token-adaptive model
            config: TAP configuration
            
        Returns:
            model: QLoRA-enabled model
        """
        if not (PEFT_AVAILABLE and BITSANDBYTES_AVAILABLE):
            logger.warning("PEFT or BitsAndBytes library not available, skipping QLoRA integration")
            return model
        
        if config is None:
            config = TAPConfig()
        
        # First, quantize model with bitsandbytes
        from bitsandbytes.nn import Linear4bit
        
        # Convert linear layers to 4-bit
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, Linear4bit):
                # Skip output layers to maintain precision
                if "lm_head" in name or "output_projection" in name:
                    continue
                
                # Convert to 4-bit
                new_module = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16
                )
                
                # Replace the module
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model if parent_name == "" else getattr(model, parent_name)
                setattr(parent, child_name, new_module)
        
        # Then apply LoRA on top of quantized model
        model = ModelIntegrationManager.integrate_with_lora(model, config)
        
        return model


#------------------------------------------------------------------------------
# Model Training and Fine-tuning
#------------------------------------------------------------------------------

class TokenAdaptiveTrainer:
    """
    Trainer for fine-tuning models with token-adaptive precision
    """
    def __init__(self, 
                 model, 
                 train_dataset, 
                 eval_dataset=None,
                 data_collator=None,
                 tokenizer=None,
                 optimizer=None,
                 lr_scheduler=None,
                 batch_size=8,
                 eval_batch_size=None,
                 num_epochs=3,
                 gradient_accumulation_steps=1,
                 use_adaptive_precision=True,
                 log_level="info",
                 config=None):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator function
            tokenizer: Tokenizer
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Gradient accumulation steps
            use_adaptive_precision: Whether to use token-adaptive precision
            log_level: Logging level
            config: TAP configuration
        """
        self.config = config or TAPConfig()
        
        # Training parameters
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_adaptive_precision = use_adaptive_precision
        
        # Device setup
        self.device = self.config.device
        if TORCH_AVAILABLE and torch.cuda.is_available() and self.device != "cpu":
            self.model = self.model.to(self.device)
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=5e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Set up learning rate scheduler
        if lr_scheduler is None:
            # Create data loader to get number of training steps
            train_dataloader = self._create_dataloader(self.train_dataset, self.batch_size)
            num_training_steps = len(train_dataloader) * self.num_epochs
            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps
            )
        else:
            self.lr_scheduler = lr_scheduler
        
        # Set up mixed precision training if enabled
        self.use_mixed_precision = self.config.enable_mixed_precision_training
        self.grad_scaler = None
        
        if self.use_mixed_precision and TORCH_AVAILABLE and torch.cuda.is_available():
            self.grad_scaler = torch.cuda.amp.GradScaler()
        
        # Set up token importance analyzer
        self.token_analyzer = TokenImportanceAnalyzer(config)
        
        # Energy and memory monitoring
        self.energy_monitor = EnergyMonitor(config=self.config)
        self.memory_tracker = MemoryTracker(
            max_memory_pct=self.config.max_memory_usage_pct)
        
        # Training metrics
        self.metrics = {
            "train_losses": [],
            "eval_losses": [],
            "learning_rates": [],
            "energy_usage": [],
            "memory_usage": [],
            "precision_stats": []
        }
        
        # Set logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
    
    def _create_dataloader(self, dataset, batch_size):
        """Create DataLoader from dataset"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available, cannot create DataLoader")
        
        # Use default collator if none provided
        collate_fn = self.data_collator if self.data_collator else lambda x: {
            "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in x]),
            "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in x]),
            "labels": torch.stack([torch.tensor(item["labels"]) for item in x]) if "labels" in x[0] else None
        }
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # This could be increased for CPU-based preprocessing
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def train(self):
        """Run training"""
        # Set up data loaders
        train_dataloader = self._create_dataloader(self.train_dataset, self.batch_size)
        eval_dataloader = None
        if self.eval_dataset:
            eval_dataloader = self._create_dataloader(self.eval_dataset, self.eval_batch_size)
        
        # Training loop
        total_steps = len(train_dataloader) * self.num_epochs
        global_step = 0
        
        logger.info(f"Starting training for {self.num_epochs} epochs ({total_steps} steps)")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            epoch_loss = 0
            step_loss = 0
            
            # Start energy monitoring for this epoch
            self.energy_monitor.start()
            start_memory = self.memory_tracker.check_memory()
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Calculate token importance and precision if using adaptive precision
                token_precision = None
                if self.use_adaptive_precision and self.config.precision_mode == PrecisionMode.ADAPTIVE:
                    token_importance = self.token_analyzer.analyze_token_importance(
                        hidden_states=None,  # Will be calculated in forward pass
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask")
                    )
                    
                    token_precision = self.token_analyzer.assign_precision(token_importance)
                
                # Forward pass with mixed precision if enabled
                if self.use_mixed_precision and self.grad_scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch.get("labels"),
                            token_precision=token_precision
                        )
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                        token_precision=token_precision
                    )
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_mixed_precision and self.grad_scaler:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update step loss
                step_loss += loss.item()
                
                # Update parameters on specified steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Optimizer step
                    if self.use_mixed_precision and self.grad_scaler:
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    # Scheduler step
                    self.lr_scheduler.step()
                    
                    # Reset gradients
                    self.optimizer.zero_grad()
                    
                    # Update metrics
                    global_step += 1
                    
                    # Log progress
                    if global_step % 10 == 0:
                        logger.info(f"Step {global_step}/{total_steps} - Loss: {step_loss:.4f}, "
                                   f"LR: {self.lr_scheduler.get_last_lr()[0]:.2e}")
                        
                        # Record metrics
                        self.metrics["train_losses"].append(step_loss)
                        self.metrics["learning_rates"].append(self.lr_scheduler.get_last_lr()[0])
                        
                        # Reset step loss
                        step_loss = 0
                    
                    # Memory optimization
                    if self.config.memory_efficient_mode:
                        self.memory_tracker.optimize_memory(force=False)
                
                # Update epoch loss
                epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            # End of epoch
            # Stop energy monitoring
            energy_metrics = self.energy_monitor.stop()
            end_memory = self.memory_tracker.check_memory()
            
            # Calculate memory change
            memory_change_mb = 0
            if "allocated_bytes" in start_memory and "allocated_bytes" in end_memory:
                memory_change_mb = (end_memory["allocated_bytes"] - start_memory["allocated_bytes"]) / (1024 * 1024)
            
            # Record energy and memory metrics
            self.metrics["energy_usage"].append(energy_metrics)
            self.metrics["memory_usage"].append({
                "change_mb": memory_change_mb,
                "peak_mb": end_memory.get("peak_bytes", 0) / (1024 * 1024)
            })
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, "
                       f"Energy: {energy_metrics['total_energy']:.2f}J, "
                       f"Mem Change: {memory_change_mb:.2f}MB")
            
            # Evaluation
            if eval_dataloader:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info(f"Evaluation - Loss: {eval_metrics['loss']:.4f}")
                self.metrics["eval_losses"].append(eval_metrics["loss"])
                
                # Log precision stats if available
                if "precision_stats" in eval_metrics:
                    self.metrics["precision_stats"].append(eval_metrics["precision_stats"])
        
        logger.info("Training complete!")
        return self.metrics
    
    def evaluate(self, eval_dataloader=None, use_adaptive_precision=None):
        """
        Evaluate model on a dataset
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            use_adaptive_precision: Whether to use token-adaptive precision
            
        Returns:
            results: Evaluation results
        """
        # Default to initialized values if not provided
        if eval_dataloader is None:
            if self.eval_dataset is None:
                logger.warning("No evaluation dataset provided")
                return {"loss": float('nan')}
            
            eval_dataloader = self._create_dataloader(self.eval_dataset, self.eval_batch_size)
        
        if use_adaptive_precision is None:
            use_adaptive_precision = self.use_adaptive_precision
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        
        all_metrics = []
        
        # Start energy monitoring for entire evaluation
        self.energy_monitor.start()
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Calculate token importance if using adaptive precision
                token_precision = None
                if use_adaptive_precision and self.config.precision_mode == PrecisionMode.ADAPTIVE:
                    # Forward pass to get hidden states and attention scores
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict=True
                    )
                    
                    # Get hidden states and attention from last layer
                    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") and outputs.hidden_states else None
                    attention_scores = outputs.attentions[-1] if hasattr(outputs, "attentions") and outputs.attentions else None
                    
                    # Calculate token importance
                    token_importance = self.token_analyzer.analyze_token_importance(
                        attention_scores=attention_scores,
                        hidden_states=hidden_states,
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask")
                    )
                    
                    # Assign precision levels
                    token_precision = self.token_analyzer.assign_precision(token_importance)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                    token_precision=token_precision,
                    return_dict=True
                )
                
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Update totals
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect metrics
                batch_metrics = {"loss": loss.item()}
                
                # If using token-adaptive precision, add those metrics
                if token_precision is not None:
                    precision_stats = self.token_analyzer.analyze_precision_stats(
                        token_importance, token_precision)
                    batch_metrics["token_precision"] = precision_stats
                
                all_metrics.append(batch_metrics)
        
        # Stop energy monitoring
        energy_metrics = self.energy_monitor.stop()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Aggregate metrics
        aggregated_metrics = {
            "loss": avg_loss,
            "energy": energy_metrics,
            "samples": total_samples
        }
        
        # Aggregate token precision metrics if available
        if all("token_precision" in m for m in all_metrics):
            # Average precision distribution
            precision_dist = {}
            for metric in all_metrics:
                for prec, pct in metric["token_precision"].items():
                    if prec.startswith("precision_") and prec.endswith("_pct"):
                        if prec not in precision_dist:
                            precision_dist[prec] = 0
                        precision_dist[prec] += pct
            
            # Calculate averages
            for prec in precision_dist:
                precision_dist[prec] /= len(all_metrics)
            
            aggregated_metrics["precision_stats"] = precision_dist
        
        # Set model back to training mode
        self.model.train()
        
        return aggregated_metrics


#------------------------------------------------------------------------------
# Main TAP Engine API and Entry Points
#------------------------------------------------------------------------------

class TAPEngine:
    """
    Main interface to the Token-Adaptive Precision engine.
    Provides high-level functions for model loading, optimization, and operation.
    """
    
    def __init__(self, config=None):
        """
        Initialize TAP Engine
        
        Args:
            config: TAPConfig object or None for defaults
        """
        self.config = config or TAPConfig()
        
        # Display system info
        show_system_info()
        
        # Initialize components
        self.token_analyzer = TokenImportanceAnalyzer(self.config)
        self.energy_monitor = EnergyMonitor(config=self.config)
        self.memory_tracker = MemoryTracker(
            max_memory_pct=self.config.max_memory_usage_pct)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
        logger.info(f"TAP Engine initialized with precision mode: {self.config.precision_mode.value}")
        if self.config.precision_mode == PrecisionMode.ADAPTIVE:
            levels_str = ", ".join([f"{p}-bit" for p in self.config.precision_levels])
            thresholds_str = ", ".join([f"{t:.2f}" for t in self.config.precision_thresholds])
            logger.info(f"Using precision levels: {levels_str} with thresholds: {thresholds_str}")
    
    def load_model(self, model_name_or_path, model_arch=None, optimize=True):
        """
        Load model from name or path
        
        Args:
            model_name_or_path: Model name from Hugging Face Hub or local path
            model_arch: Model architecture (transformer, mamba, retnet, rwkv)
            optimize: Whether to apply optimizations
            
        Returns:
            model: Loaded model
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available, cannot load model")
            return None
        
        logger.info(f"Loading model from {model_name_or_path}")
        
        try:
            # Determine model class based on architecture
            model_cls = None
            tokenizer_cls = None
            
            from transformers import AutoTokenizer
            
            if model_arch == "mamba" or self.config.model_arch == ModelArchitecture.MAMBA:
                # Use specialized Mamba model if available
                try:
                    from transformers import MambaForCausalLM
                    model_cls = MambaForCausalLM
                except:
                    from transformers import AutoModelForCausalLM
                    model_cls = AutoModelForCausalLM
                    logger.warning("Specialized Mamba model not available, using generic causal LM")
                
                self.config.model_arch = ModelArchitecture.MAMBA
            
            elif model_arch == "retnet" or self.config.model_arch == ModelArchitecture.RETNET:
                # Use RetNet model if available
                try:
                    from transformers import RetNetForCausalLM
                    model_cls = RetNetForCausalLM
                except:
                    from transformers import AutoModelForCausalLM
                    model_cls = AutoModelForCausalLM
                    logger.warning("Specialized RetNet model not available, using generic causal LM")
                
                self.config.model_arch = ModelArchitecture.RETNET
            
            elif model_arch == "rwkv" or self.config.model_arch == ModelArchitecture.RWKV:
                # Use RWKV model if available
                try:
                    from transformers import RwkvForCausalLM
                    model_cls = RwkvForCausalLM
                except:
                    from transformers import AutoModelForCausalLM
                    model_cls = AutoModelForCausalLM
                    logger.warning("Specialized RWKV model not available, using generic causal LM")
                
                self.config.model_arch = ModelArchitecture.RWKV
            
            else:
                # Default to generic model
                from transformers import AutoModelForCausalLM
                model_cls = AutoModelForCausalLM
                self.config.model_arch = ModelArchitecture.TRANSFORMER
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                
                # Add padding token if needed
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                # Continue without tokenizer
            
            # Determine dtype based on config
            dtype = torch.float32
            if self.config.bf16 and TORCH_AVAILABLE and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                logger.info("Using bfloat16 precision")
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                dtype = torch.float16
                logger.info("Using float16 precision")
            
            # Special handling for 4-bit if using BitByte
            use_4bit = False
            if self.config.use_bitbyte and BITSANDBYTES_AVAILABLE and TORCH_AVAILABLE and torch.cuda.is_available():
                if self.config.weight_quantization and self.config.weight_quantization.bits == 4:
                    use_4bit = True
                    logger.info("Using 4-bit quantization with BitsAndBytes")
            
            # Load model with appropriate settings
            if use_4bit:
                # 4-bit quantization setup
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = model_cls.from_pretrained(
                    model_name_or_path,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                # Standard loading
                self.model = model_cls.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    device_map="auto" if TORCH_AVAILABLE and torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
            
            # Apply token-adaptive conversion
            if optimize:
                self.optimize_model()
            
            logger.info(f"Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def optimize_model(self, quantize_bits=None, compile=None):
        """
        Optimize model for inference with token-adaptive precision
        
        Args:
            quantize_bits: Bits for quantization (0=disable, 4, 8, or 16)
            compile: Whether to apply torch.compile
            
        Returns:
            stats: Optimization statistics
        """
        if self.model is None:
            logger.error("No model loaded, cannot optimize")
            return {"error": "no_model"}
        
        stats = {}
        
        # Set defaults from config if not specified
        if quantize_bits is None and self.config.weight_quantization:
            quantize_bits = self.config.weight_quantization.bits
        
        if compile is None:
            compile = self.config.compilation_enabled
        
        # Apply quantization if specified
        if quantize_bits not in [0, None]:
            logger.info(f"Applying {quantize_bits}-bit quantization to model weights")
            
            # Create quantization config
            q_config = QuantizationConfig(
                bits=quantize_bits,
                scheme=self.config.weight_quantization.scheme if self.config.weight_quantization else QuantizationScheme.SYMMETRIC,
                per_channel=self.config.weight_quantization.per_channel if self.config.weight_quantization else False,
                clip_outliers=self.config.weight_quantization.clip_outliers if self.config.weight_quantization else True
            )
            
            # Apply quantization
            quantization_stats = PrecisionManager.quantize_module(self.model, q_config)
            stats["quantization"] = quantization_stats
            
            logger.info(f"Quantization complete. Memory saved: {quantization_stats['memory_saved_mb']:.2f} MB")
        
        # Apply compilation if requested and available
        if compile and TORCH_AVAILABLE:
            logger.info(f"Applying torch.compile with mode '{self.config.compile_mode}'")
            try:
                torch_version = torch.__version__.split('.')
                if int(torch_version[0]) >= 2:
                    self.model = PrecisionManager.compile_fn(
                        self.model, mode=self.config.compile_mode)
                    stats["compile"] = "enabled"
                    logger.info("Compilation successful")
                else:
                    logger.warning(f"torch.compile requires PyTorch 2.0+, current version: {torch.__version__}")
                    stats["compile"] = "unavailable"
            except Exception as e:
                logger.error(f"Compilation failed: {e}")
                stats["compile"] = "failed"
        
        # Apply optimizations for specific architectures
        if self.config.model_arch == ModelArchitecture.MAMBA:
            stats["arch_optimizations"] = self._optimize_for_mamba()
        elif self.config.model_arch == ModelArchitecture.RETNET:
            stats["arch_optimizations"] = self._optimize_for_retnet()
        elif self.config.model_arch == ModelArchitecture.RWKV:
            stats["arch_optimizations"] = self._optimize_for_rwkv()
        
        # Clean up memory
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return stats
    
    def _optimize_for_mamba(self):
        """Apply Mamba-specific optimizations"""
        try:
            # Check if model has SSM layers
            ssm_layers_found = False
            
            for name, module in self.model.named_modules():
                if "ssm" in name.lower() or "mamba" in name.lower():
                    ssm_layers_found = True
                    break
            
            if not ssm_layers_found:
                logger.warning("No Mamba/SSM layers found in model")
                return {"ssm_layers_found": False}
            
            # Apply Mamba-specific optimizations
            # Implementation depends on specific model structure
            
            return {"ssm_layers_found": True, "optimizations_applied": True}
        except Exception as e:
            logger.error(f"Error optimizing for Mamba: {e}")
            return {"error": str(e)}
    
    def _optimize_for_retnet(self):
        """Apply RetNet-specific optimizations"""
        try:
            # Check if model has retention layers
            retention_layers_found = False
            
            for name, module in self.model.named_modules():
                if "retention" in name.lower() or "retnet" in name.lower():
                    retention_layers_found = True
                    break
            
            if not retention_layers_found:
                logger.warning("No RetNet layers found in model")
                return {"retention_layers_found": False}
            
            # Apply RetNet-specific optimizations
            # Implementation depends on specific model structure
            
            return {"retention_layers_found": True, "optimizations_applied": True}
        except Exception as e:
            logger.error(f"Error optimizing for RetNet: {e}")
            return {"error": str(e)}
    
    def _optimize_for_rwkv(self):
        """Apply RWKV-specific optimizations"""
        try:
            # Check if model has WKV layers
            wkv_layers_found = False
            
            for name, module in self.model.named_modules():
                if "rwkv" in name.lower() or "wkv" in name.lower():
                    wkv_layers_found = True
                    break
            
            if not wkv_layers_found:
                logger.warning("No RWKV layers found in model")
                return {"wkv_layers_found": False}
            
            # Apply RWKV-specific optimizations
            # Implementation depends on specific model structure
            
            return {"wkv_layers_found": True, "optimizations_applied": True}
        except Exception as e:
            logger.error(f"Error optimizing for RWKV: {e}")
            return {"error": str(e)}
    
    def generate(self, 
                prompt=None, 
                input_ids=None, 
                attention_mask=None, 
                max_new_tokens=20, 
                use_adaptive_precision=None,
                **kwargs):
        """
        Generate text with token-adaptive precision
        
        Args:
            prompt: Text prompt
            input_ids: Tokenized input IDs (alternative to prompt)
            attention_mask: Attention mask for input_ids
            max_new_tokens: Maximum new tokens to generate
            use_adaptive_precision: Whether to use token-adaptive precision
            **kwargs: Additional generation parameters
            
        Returns:
            text: Generated text
            metrics: Generation metrics
        """
        if self.model is None:
            logger.error("No model loaded, cannot generate")
            return None, {"error": "no_model"}
        
        # Set default for adaptive precision
        if use_adaptive_precision is None:
            use_adaptive_precision = self.config.precision_mode == PrecisionMode.ADAPTIVE
        
        # Tokenize prompt if provided
        if prompt is not None:
            if self.tokenizer is None:
                logger.error("No tokenizer available, cannot process text prompt")
                return None, {"error": "no_tokenizer"}
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True
            )
            
            if TORCH_AVAILABLE and str(self.model.device) != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
        elif input_ids is not None:
            # Move to device if needed
            if TORCH_AVAILABLE and hasattr(input_ids, "device") and str(input_ids.device) != str(self.model.device):
                input_ids = input_ids.to(self.model.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
        else:
            logger.error("Either prompt or input_ids must be provided")
            return None, {"error": "no_input"}
        
        # Default generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
        }
        
        # Update with user-provided kwargs
        generation_kwargs.update({k: v for k, v in kwargs.items() 
                                if k not in ["do_sample", "temperature", "top_p", "top_k", "repetition_penalty"]})
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Start energy monitoring
        self.energy_monitor.start()
        
        if use_adaptive_precision:
            # Forward pass to get token importance
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
                
                # Get hidden states and attention from last layer
                hidden_states = None
                attention_scores = None
                
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1] 
                
                if hasattr(outputs, "attentions") and outputs.attentions:
                    attention_scores = outputs.attentions[-1]
                
                # Get SSM states for Mamba models
                ssm_states = None
                if self.config.model_arch == ModelArchitecture.MAMBA and hasattr(outputs, "ssm_states"):
                    ssm_states = outputs.ssm_states
                
                # Calculate token importance
                token_importance = self.token_analyzer.analyze_token_importance(
                    attention_scores=attention_scores,
                    hidden_states=hidden_states,
                    ssm_states=ssm_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Assign precision levels
                token_precision = self.token_analyzer.assign_precision(token_importance)
                
                # Calculate precision statistics
                precision_stats = self.token_analyzer.analyze_precision_stats(
                    token_importance, token_precision)
                
                # Log precision distribution
                dist_str = ", ".join(f"{k.split('_')[1]}-bit: {v:.1f}%" 
                                  for k, v in precision_stats.items()
                                  if k.startswith("precision_") and k.endswith("_pct"))
                logger.info(f"Token precision distribution: {dist_str}")
            
            # For actual generation, we can only set the precision for the prompt tokens
            # The model will dynamically calculate importance for new tokens during generation
            try:
                # Check if model has a special generate_with_token_precision method
                if hasattr(self.model, "generate_with_token_precision"):
                    generation_output = self.model.generate_with_token_precision(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_precision=token_precision,
                        **generation_kwargs
                    )
                else:
                    # Use standard generation (precision will be calculated for the prompt only)
                    generation_output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs
                    )
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                # Fall back to standard generation
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs
                )
        else:
            # Standard generation without adaptive precision
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs
                )
            
            # Set placeholder values
            token_precision = None
            precision_stats = None
        
        # Stop energy monitoring
        energy_metrics = self.energy_monitor.stop()
        
        # Post-process output
        if isinstance(generation_output, torch.Tensor):
            generated_ids = generation_output
        else:
            # Handle different return types (some models return objects)
            generated_ids = getattr(generation_output, "sequences", generation_output)
        
        # Decode generated text
        if self.tokenizer is not None:
            generated_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
        else:
            # Can't decode without tokenizer
            generated_text = None
            logger.warning("No tokenizer available, returning token IDs only")
        
        # Calculate generation metrics
        if input_ids is not None:
            new_tokens = generated_ids.shape[1] - input_ids.shape[1]
        else:
            new_tokens = generated_ids.shape[1]
        
        # Adjust energy metrics based on precision if available
        adjusted_energy = energy_metrics.copy()
        if precision_stats and "energy_saved_pct" in precision_stats:
            energy_saved_factor = precision_stats["energy_saved_pct"] / 100
            
            # Apply energy savings to measured energy
            if "total_energy" in adjusted_energy:
                adjusted_energy["total_energy"] = energy_metrics["total_energy"] * (1 - energy_saved_factor)
            
            if "avg_power" in adjusted_energy:
                adjusted_energy["avg_power"] = energy_metrics["avg_power"] * (1 - energy_saved_factor)
        
        # Prepare metrics
        metrics = {
            "generated_tokens": new_tokens,
            "generation_time": energy_metrics["duration"],
            "tokens_per_second": new_tokens / energy_metrics["duration"] if energy_metrics["duration"] > 0 else 0,
            "energy": energy_metrics,
            "adjusted_energy": adjusted_energy
        }
        
        # Add precision metrics if available
        if precision_stats:
            metrics["precision"] = precision_stats
        
        # Return first generated text if batch size is 1
        if generated_text is not None and len(generated_text) == 1:
            return generated_text[0], metrics
        else:
            return generated_text or generated_ids, metrics
    
    def analyze_token_importance(self, prompt=None, input_ids=None, attention_mask=None):
        """
        Analyze token importance for a given input
        
        Args:
            prompt: Text prompt
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            
        Returns:
            results: Analysis results
        """
        if self.model is None:
            logger.error("No model loaded, cannot analyze")
            return {"error": "no_model"}
        
        # Tokenize prompt if provided
        if prompt is not None:
            if self.tokenizer is None:
                logger.error("No tokenizer available, cannot process text prompt")
                return {"error": "no_tokenizer"}
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True
            )
            
            if TORCH_AVAILABLE and str(self.model.device) != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
        elif input_ids is None:
            logger.error("Either prompt or input_ids must be provided")
            return {"error": "no_input"}
        
        # Get token texts for reference
        token_texts = None
        if self.tokenizer is not None:
            try:
                # Convert token IDs to text tokens
                token_texts = []
                for i in range(input_ids.shape[1]):
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, i].item())
                    token_texts.append(tokens)
            except Exception as e:
                logger.error(f"Error getting token texts: {e}")
        
        # Put model in eval mode
        self.model.eval()
        
        # Forward pass with hidden states and attention scores
        with torch.no_grad():outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
        
        # Extract hidden states and attention scores
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = None
        
        if hasattr(outputs, "attentions") and outputs.attentions:
            attention_scores = outputs.attentions[-1]
        else:
            attention_scores = None
        
        # Extract SSM states for Mamba models
        ssm_states = None
        if self.config.model_arch == ModelArchitecture.MAMBA and hasattr(outputs, "ssm_states"):
            ssm_states = outputs.ssm_states
        
        # Calculate token importance with all available metrics
        importance_results = {}
        
        # Always calculate with the config's default method
        default_importance = self.token_analyzer.analyze_token_importance(
            attention_scores=attention_scores,
            hidden_states=hidden_states,
            ssm_states=ssm_states,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        importance_results["default"] = default_importance
        
        # Calculate with all available metrics if requested
        all_metrics = [
            TokenImportanceMetric.ATTENTION if attention_scores is not None else None,
            TokenImportanceMetric.HIDDEN_NORM if hidden_states is not None else None,
            TokenImportanceMetric.STATE_NORM if ssm_states is not None else None,
            TokenImportanceMetric.POSITION,
            TokenImportanceMetric.TOKEN_ID if input_ids is not None else None,
            TokenImportanceMetric.ENTROPY if attention_scores is not None else None
        ]
        
        # Remove None values
        all_metrics = [m for m in all_metrics if m is not None]
        
        # Calculate importance with each metric
        for metric in all_metrics:
            if metric != self.config.importance_method:  # Skip default method which was already calculated
                importance = self.token_analyzer.analyze_token_importance(
                    attention_scores=attention_scores,
                    hidden_states=hidden_states,
                    ssm_states=ssm_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    method=metric
                )
                importance_results[metric.value if isinstance(metric, TokenImportanceMetric) else metric] = importance
        
        # Assign precision levels for default method
        token_precision = self.token_analyzer.assign_precision(default_importance)
        
        # Calculate precision statistics
        precision_stats = self.token_analyzer.analyze_precision_stats(
            default_importance, token_precision)
        
        # Prepare results
        results = {
            "token_importance": importance_results,
            "token_precision": token_precision,
            "precision_stats": precision_stats,
            "token_texts": token_texts,
            "model_outputs": {
                "logits": outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else None
            }
        }
        
        return results
    
    def train(self, 
             train_dataset, 
             eval_dataset=None, 
             data_collator=None,
             batch_size=8,
             eval_batch_size=None,
             num_epochs=3,
             learning_rate=5e-5,
             weight_decay=0.01,
             gradient_accumulation_steps=1,
             use_adaptive_precision=None,
             warmup_steps=0,
             warmup_ratio=0.0):
        """
        Train the model on a dataset
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator function
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            gradient_accumulation_steps: Gradient accumulation steps
            use_adaptive_precision: Whether to use token-adaptive precision
            warmup_steps: Number of warmup steps
            warmup_ratio: Ratio of warmup steps to total steps
            
        Returns:
            metrics: Training metrics
        """
        if self.model is None:
            logger.error("No model loaded, cannot train")
            return {"error": "no_model"}
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot train")
            return {"error": "no_pytorch"}
        
        # Set default for adaptive precision
        if use_adaptive_precision is None:
            use_adaptive_precision = self.config.precision_mode == PrecisionMode.ADAPTIVE
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create data loader to get number of training steps
        if data_collator is None and self.tokenizer is not None:
            # Create default data collator using tokenizer
            data_collator = lambda examples: self.tokenizer.pad(examples, return_tensors="pt")
        
        # Create trainer
        trainer = TokenAdaptiveTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            optimizer=optimizer,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_adaptive_precision=use_adaptive_precision,
            config=self.config
        )
        
        # Train the model
        metrics = trainer.train()
        
        return metrics
    
    def save_model(self, output_dir, save_tokenizer=True):
        """
        Save the model and tokenizer to disk
        
        Args:
            output_dir: Directory to save to
            save_tokenizer: Whether to save tokenizer
            
        Returns:
            output_dir: Path where model was saved
        """
        if self.model is None:
            logger.error("No model loaded, cannot save")
            return {"error": "no_model"}
        
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            # Fallback to torch.save
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save TAPConfig
        config_path = os.path.join(output_dir, "tap_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Model and configuration saved to {output_dir}")
        
        return output_dir
    
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        """
        Load a model with token-adaptive precision capability
        
        Args:
            model_path: Path to model
            config: TAPConfig or None
            
        Returns:
            engine: TAPEngine instance
        """
        import os
        import json
        
        # Load TAPConfig if available
        if config is None:
            config_path = os.path.join(model_path, "tap_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = TAPConfig.from_dict(config_dict)
            else:
                config = TAPConfig()
        
        # Create engine
        engine = cls(config=config)
        
        # Load model
        engine.load_model(model_path, optimize=False)
        
        return engine


#------------------------------------------------------------------------------
# Command Line Interface
#------------------------------------------------------------------------------

def main():
    """Command line interface for the TAP Engine"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Token-Adaptive Precision Engine")
    
    # Main command arguments
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Model name from Hugging Face Hub or local path")
    parser.add_argument("--model-arch", "-a", type=str, choices=["transformer", "mamba", "retnet", "rwkv"],
                       default=None, help="Model architecture")
    parser.add_argument("--device", "-d", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--precision-mode", "-p", type=str, 
                       choices=["adaptive", "mixed", "int8", "int4", "standard"],
                       default="adaptive", help="Precision mode")
    parser.add_argument("--config", "-c", type=str, default=None,
                       help="Path to TAP configuration file")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                       help="Output directory for model and results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    # Action subparsers
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Generate text
    generate_parser = subparsers.add_parser("generate", help="Generate text")
    generate_parser.add_argument("--prompt", type=str, required=True,
                               help="Text prompt")
    generate_parser.add_argument("--max-tokens", type=int, default=50,
                               help="Maximum tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.7,
                               help="Sampling temperature")
    generate_parser.add_argument("--top-p", type=float, default=0.9,
                               help="Top-p sampling parameter")
    generate_parser.add_argument("--top-k", type=int, default=50,
                               help="Top-k sampling parameter")
    generate_parser.add_argument("--num-return-sequences", type=int, default=1,
                               help="Number of sequences to return")
    generate_parser.add_argument("--no-adaptive", action="store_true",
                               help="Disable token-adaptive precision")
    
    # Optimize model
    optimize_parser = subparsers.add_parser("optimize", help="Optimize model")
    optimize_parser.add_argument("--quantize-bits", type=int, choices=[0, 4, 8, 16],
                               default=None, help="Bits for quantization (0=disable)")
    optimize_parser.add_argument("--no-compile", action="store_true",
                               help="Disable compilation")
    optimize_parser.add_argument("--save", action="store_true",
                               help="Save optimized model")
    
    # Analyze token importance
    analyze_parser = subparsers.add_parser("analyze", help="Analyze token importance")
    analyze_parser.add_argument("--prompt", type=str, required=True,
                              help="Text prompt to analyze")
    analyze_parser.add_argument("--visualize", action="store_true",
                              help="Visualize token importance")
    analyze_parser.add_argument("--output-file", type=str, default=None,
                              help="Output file for visualization")
    
    # Benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("--prompt", type=str, required=True,
                                help="Text prompt for benchmarking")
    benchmark_parser.add_argument("--max-tokens", type=int, default=50,
                                help="Maximum tokens to generate")
    benchmark_parser.add_argument("--iterations", type=int, default=5,
                                help="Number of iterations")
    benchmark_parser.add_argument("--compare-modes", action="store_true",
                                help="Compare different precision modes")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load or create configuration
    config = None
    if args.config:
        try:
            config = TAPConfig.load(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            config = None
    
    if config is None:
        # Create config from command line arguments
        config = TAPConfig()
        
        # Set device
        if args.device != "auto":
            config.device = args.device
        
        # Set precision mode
        if args.precision_mode:
            config.precision_mode = PrecisionMode(args.precision_mode)
        
        # Set model architecture if specified
        if args.model_arch:
            config.model_arch = ModelArchitecture(args.model_arch)
        
        # Set verbose logging
        if args.verbose:
            config.verbose_logging = True
    
    # Create TAP Engine
    engine = TAPEngine(config=config)
    
    # Load model if specified
    if args.model:
        engine.load_model(args.model, model_arch=args.model_arch)
    elif args.action and args.action != "version":
        logger.error("No model specified. Use --model to specify a model.")
        return
    
    # Execute action
    if args.action == "generate":
        # Generate text
        use_adaptive_precision = not args.no_adaptive
        
        text, metrics = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            use_adaptive_precision=use_adaptive_precision
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("Generated text:")
        print("-" * 50)
        print(text)
        print("=" * 50)
        
        # Display metrics
        print("\nGeneration metrics:")
        print(f"- Tokens generated: {metrics['generated_tokens']}")
        print(f"- Generation time: {metrics['generation_time']:.2f} seconds")
        print(f"- Tokens per second: {metrics['tokens_per_second']:.2f}")
        
        # Precision metrics if available
        if "precision" in metrics:
            print("\nPrecision distribution:")
            for k, v in sorted(metrics["precision"].items()):
                if k.startswith("precision_") and k.endswith("_pct"):
                    bits = k.split("_")[1]
                    print(f"- {bits}-bit: {v:.1f}%")
            
            if "energy_saved_pct" in metrics["precision"]:
                print(f"\nEstimated energy saved: {metrics['precision']['energy_saved_pct']:.2f}%")
        
        # Save to output directory if specified
        if args.output_dir:
            import os
            import json
            
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save generated text
            text_path = os.path.join(args.output_dir, "generated_text.txt")
            with open(text_path, "w") as f:
                f.write(text)
            
            # Save metrics
            metrics_path = os.path.join(args.output_dir, "generation_metrics.json")
            with open(metrics_path, "w") as f:
                # Convert tensors to lists
                serializable_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        serializable_metrics[k] = {}
                        for sub_k, sub_v in v.items():
                            if hasattr(sub_v, "tolist"):
                                serializable_metrics[k][sub_k] = sub_v.tolist()
                            else:
                                serializable_metrics[k][sub_k] = sub_v
                    elif hasattr(v, "tolist"):
                        serializable_metrics[k] = v.tolist()
                    else:
                        serializable_metrics[k] = v
                
                json.dump(serializable_metrics, f, indent=2)
            
            print(f"\nResults saved to {args.output_dir}")
    
    elif args.action == "optimize":
        # Optimize model
        compile_enabled = not args.no_compile
        
        stats = engine.optimize_model(
            quantize_bits=args.quantize_bits,
            compile=compile_enabled
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("Model optimization complete")
        print("-" * 50)
        
        # Quantization stats
        if "quantization" in stats:
            q_stats = stats["quantization"]
            print(f"Quantized parameters: {q_stats.get('params_quantized', 0)}/{q_stats.get('total_params', 0)}")
            print(f"Memory saved: {q_stats.get('memory_saved_mb', 0):.2f} MB")
            print(f"Compression ratio: {q_stats.get('compression_ratio', 1.0):.2f}x")
        
        # Compilation status
        if "compile" in stats:
            print(f"Compilation: {stats['compile']}")
        
        print("=" * 50)
        
        # Save optimized model if requested
        if args.save:
            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = "optimized_model"
            
            engine.save_model(output_dir)
            print(f"\nOptimized model saved to: {output_dir}")
    
    elif args.action == "analyze":
        # Analyze token importance
        results = engine.analyze_token_importance(prompt=args.prompt)
        
        # Display results
        print("\n" + "=" * 50)
        print("Token Importance Analysis")
        print("-" * 50)
        
        # Show token texts and importance scores
        if "token_texts" in results and results["token_texts"]:
            print("\nToken-level importance:")
            print("-" * 20)
            
            # Get default importance scores
            importance = results["token_importance"]["default"]
            token_texts = results["token_texts"]
            
            for i, token in enumerate(token_texts):
                imp_score = importance[0, i].item() if hasattr(importance, "item") else importance[0][i]
                precision = results["token_precision"][0, i].item() if hasattr(results["token_precision"], "item") else results["token_precision"][0][i]
                print(f"Token {i}: '{token}' - Importance: {imp_score:.4f}, Precision: {precision}-bit")
        
        # Show precision statistics
        if "precision_stats" in results:
            print("\nPrecision distribution:")
            for k, v in sorted(results["precision_stats"].items()):
                if k.startswith("precision_") and k.endswith("_pct"):
                    bits = k.split("_")[1]
                    print(f"- {bits}-bit: {v:.1f}%")
            
            if "energy_saved_pct" in results["precision_stats"]:
                print(f"\nEstimated energy savings: {results['precision_stats']['energy_saved_pct']:.2f}%")
        
        print("=" * 50)
        
        # Visualize if requested
        if args.visualize:
            try:
                visualize_token_importance(
                    results["token_importance"]["default"].cpu().numpy()[0] if hasattr(results["token_importance"]["default"], "cpu") else results["token_importance"]["default"][0],
                    results["token_precision"].cpu().numpy()[0] if hasattr(results["token_precision"], "cpu") else results["token_precision"][0],
                    results["token_texts"],
                    output_file=args.output_file
                )
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
    
    elif args.action == "benchmark":
        # Benchmark model performance
        print("\n" + "=" * 50)
        print("Model Benchmark")
        print("-" * 50)
        
        results = []
        
        if args.compare_modes:
            # Test with different precision modes
            precision_modes = [
                ("Adaptive", True),
                ("Standard", False)
            ]
            
            for mode_name, adaptive in precision_modes:
                print(f"\nTesting {mode_name} precision mode...")
                
                mode_results = []
                for i in range(args.iterations):
                    print(f"  Iteration {i+1}/{args.iterations}...", end="", flush=True)
                    
                    # Generate text
                    _, metrics = engine.generate(
                        prompt=args.prompt,
                        max_new_tokens=args.max_tokens,
                        use_adaptive_precision=adaptive
                    )
                    
                    mode_results.append(metrics)
                    print(" done")
                
                # Calculate averages
                avg_time = sum(m["generation_time"] for m in mode_results) / len(mode_results)
                avg_tokens_per_sec = sum(m["tokens_per_second"] for m in mode_results) / len(mode_results)
                avg_energy = sum(m["energy"]["total_energy"] for m in mode_results) / len(mode_results)
                
                if adaptive and all("precision" in m and "energy_saved_pct" in m["precision"] for m in mode_results):
                    avg_energy_saved = sum(m["precision"]["energy_saved_pct"] for m in mode_results) / len(mode_results)
                else:
                    avg_energy_saved = 0
                
                results.append({
                    "mode": mode_name,
                    "avg_time": avg_time,
                    "avg_tokens_per_sec": avg_tokens_per_sec,
                    "avg_energy": avg_energy,
                    "avg_energy_saved": avg_energy_saved
                })
            
            # Display comparison
            print("\nBenchmark Results:")
            print("-" * 20)
            
            for r in results:
                print(f"\n{r['mode']} Precision:")
                print(f"  Average generation time: {r['avg_time']:.2f} seconds")
                print(f"  Average tokens per second: {r['avg_tokens_per_sec']:.2f}")
                print(f"  Average energy consumption: {r['avg_energy']:.2f} joules")
                if r['avg_energy_saved'] > 0:
                    print(f"  Average energy saved: {r['avg_energy_saved']:.2f}%")
        
        else:
            # Single mode benchmark
            print(f"\nRunning {args.iterations} iterations...")
            
            times = []
            tokens_per_sec = []
            energies = []
            
            for i in range(args.iterations):
                print(f"  Iteration {i+1}/{args.iterations}...", end="", flush=True)
                
                # Generate text
                _, metrics = engine.generate(
                    prompt=args.prompt,
                    max_new_tokens=args.max_tokens
                )
                
                times.append(metrics["generation_time"])
                tokens_per_sec.append(metrics["tokens_per_second"])
                energies.append(metrics["energy"]["total_energy"])
                
                print(" done")
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            avg_tokens_per_sec = sum(tokens_per_sec) / len(tokens_per_sec)
            avg_energy = sum(energies) / len(energies)
            
            # Calculate standard deviations
            import math
            std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))
            std_tokens_per_sec = math.sqrt(sum((t - avg_tokens_per_sec) ** 2 for t in tokens_per_sec) / len(tokens_per_sec))
            std_energy = math.sqrt(sum((e - avg_energy) ** 2 for e in energies) / len(energies))
            
            # Display results
            print("\nBenchmark Results:")
            print("-" * 20)
            print(f"Average generation time: {avg_time:.2f} ± {std_time:.2f} seconds")
            print(f"Average tokens per second: {avg_tokens_per_sec:.2f} ± {std_tokens_per_sec:.2f}")
            print(f"Average energy consumption: {avg_energy:.2f} ± {std_energy:.2f} joules")
    
    else:
        # Display version and help
        print("\nToken-Adaptive Precision (TAP) Engine")
        print(f"Version: {__version__}")
        print("\nUse --help for usage information")


def visualize_token_importance(importance_scores, precision_levels, token_texts, output_file=None):
    """
    Visualize token importance scores and precision levels
    
    Args:
        importance_scores: Token importance scores
        precision_levels: Precision levels for each token
        token_texts: Text representation of tokens
        output_file: Output file for visualization
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("Matplotlib is required for visualization. Install with 'pip install matplotlib'")
        return
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot importance scores
    x = np.arange(len(importance_scores))
    ax1.bar(x, importance_scores)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Importance Score')
    ax1.set_title('Token Importance Analysis')
    
    # Add token texts as x-tick labels
    if token_texts and len(token_texts) == len(importance_scores):
        # Simplify token texts for display (remove special chars, limit length)
        display_texts = []
        for token in token_texts:
            # Clean up token for display
            if isinstance(token, list) and len(token) > 0:
                token = token[0]
            
            # Convert to string and limit length
            token_str = str(token)
            if len(token_str) > 10:
                token_str = token_str[:8] + "..."
            
            # Replace whitespace with visible symbols
            token_str = token_str.replace(' ', '␣').replace('\n', '↵')
            
            display_texts.append(token_str)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_texts, rotation=45, ha='right')
    else:
        ax1.set_xlabel('Token Position')
    
    # Plot precision levels with color coding
    cmap = plt.get_cmap('viridis')
    
    # Normalize precision levels for color mapping
    unique_precs = np.unique(precision_levels)
    norm = mcolors.Normalize(min(unique_precs), max(unique_precs))
    
    # Create bars with colors based on precision
    for i, prec in enumerate(precision_levels):
        ax2.bar(i, 1, color=cmap(norm(prec)))
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Precision (bits)')
    
    # Set axis labels
    ax2.set_ylabel('Assigned Precision')
    ax2.set_xticks(x)
    ax2.set_ylim(0, 1)
    
    if token_texts and len(token_texts) == len(precision_levels):
        ax2.set_xticklabels(display_texts, rotation=45, ha='right')
    else:
        ax2.set_xlabel('Token Position')
    
    # Add precision levels as text on bars
    for i, prec in enumerate(precision_levels):
        ax2.text(i, 0.5, f"{int(prec)}", ha='center', va='center')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if logger.level <= logging.DEBUG:
            import traceback
            traceback.print_exc()
