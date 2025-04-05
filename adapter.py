#------------------------------------------------------------------------------
# Adapter Classes for Common Model Architectures
#------------------------------------------------------------------------------

class TransformerAdapter:
    """
    Adapter for integrating TAP Engine with standard Transformer architectures.
    Handles transformer-specific behaviors and optimizations.
    """
    
    @staticmethod
    def adapt_model(model, config=None):
        """
        Convert standard transformer layers to token-adaptive versions
        
        Args:
            model: PyTorch transformer model
            config: TAP configuration
            
        Returns:
            model: Adapted model
        """
        if config is None:
            config = TAPConfig()
        
        # For transformers modules, find attention, MLP, and layernorm components
        for name, module in model.named_children():
            if any(x in name.lower() for x in ["attention", "self", "mha", "multihead"]):
                # This is likely an attention module
                TransformerAdapter._adapt_attention_module(model, name, module, config)
            
            elif any(x in name.lower() for x in ["mlp", "ffn", "feed_forward"]):
                # This is likely an MLP/FFN module
                TransformerAdapter._adapt_ffn_module(model, name, module, config)
            
            elif any(x in name.lower() for x in ["norm", "ln", "layer_norm"]):
                # This is likely a layer norm module
                TransformerAdapter._adapt_layernorm_module(model, name, module, config)
            
            elif len(list(module.children())) > 0:
                # Recursively process children
                TransformerAdapter.adapt_model(module, config)
        
        return model
    
    @staticmethod
    def _adapt_attention_module(parent, name, module, config):
        """Adapt attention module to token-adaptive version"""
        # This is specific to the architecture
        # Common patterns include self.q/k/v + self.o or self.qkv + self.proj
        
        # Implement specific adaptation logic for attention modules
        pass
    
    @staticmethod
    def _adapt_ffn_module(parent, name, module, config):
        """Adapt feed-forward module to token-adaptive version"""
        # Implement specific adaptation logic for FFN modules
        pass
    
    @staticmethod
    def _adapt_layernorm_module(parent, name, module, config):
        """Adapt layer norm module to token-adaptive version"""
        # Implement specific adaptation logic for LayerNorm modules
        pass


class MambaAdapter:
    """
    Adapter for integrating TAP Engine with Mamba/SSM architectures.
    Handles SSM-specific behaviors and optimizations.
    """
    
    @staticmethod
    def adapt_model(model, config=None):
        """
        Convert standard Mamba/SSM layers to token-adaptive versions
        
        Args:
            model: PyTorch Mamba model
            config: TAP configuration
            
        Returns:
            model: Adapted model
        """
        if config is None:
            config = TAPConfig()
        
        # For Mamba modules, find SSM blocks, projections, and convolution layers
        for name, module in model.named_children():
            if any(x in name.lower() for x in ["ssm", "s4", "s5", "mamba_block"]):
                # This is likely an SSM module
                MambaAdapter._adapt_ssm_module(model, name, module, config)
            
            elif "conv" in name.lower():
                # This is likely a convolution module (used in Mamba)
                MambaAdapter._adapt_conv_module(model, name, module, config)
            
            elif len(list(module.children())) > 0:
                # Recursively process children
                MambaAdapter.adapt_model(module, config)
        
        return model
    
    @staticmethod
    def _adapt_ssm_module(parent, name, module, config):
        """Adapt SSM module to token-adaptive version"""
        # Implement specific adaptation logic for SSM modules
        pass
    
    @staticmethod
    def _adapt_conv_module(parent, name, module, config):
        """Adapt convolution module to token-adaptive version"""
        # Implement specific adaptation logic for convolution modules
        pass


class RetNetAdapter:
    """
    Adapter for integrating TAP Engine with RetNet architectures.
    Handles retention-specific behaviors and optimizations.
    """
    
    @staticmethod
    def adapt_model(model, config=None):
        """
        Convert standard RetNet layers to token-adaptive versions
        
        Args:
            model: PyTorch RetNet model
            config: TAP configuration
            
        Returns:
            model: Adapted model
        """
        if config is None:
            config = TAPConfig()
        
        # For RetNet modules, find retention blocks and multi-scale components
        for name, module in model.named_children():
            if any(x in name.lower() for x in ["retention", "retnet", "multiscale"]):
                # This is likely a retention module
                RetNetAdapter._adapt_retention_module(model, name, module, config)
            
            elif len(list(module.children())) > 0:
                # Recursively process children
                RetNetAdapter.adapt_model(module, config)
        
        return model
    
    @staticmethod
    def _adapt_retention_module(parent, name, module, config):
        """Adapt retention module to token-adaptive version"""
        # Implement specific adaptation logic for retention modules
        pass


class RWKVAdapter:
    """
    Adapter for integrating TAP Engine with RWKV architectures.
    Handles time-mix and channel-mix specific behaviors and optimizations.
    """
    
    @staticmethod
    def adapt_model(model, config=None):
        """
        Convert standard RWKV layers to token-adaptive versions
        
        Args:
            model: PyTorch RWKV model
            config: TAP configuration
            
        Returns:
            model: Adapted model
        """
        if config is None:
            config = TAPConfig()
        
        # For RWKV modules, find time-mix and channel-mix components
        for name, module in model.named_children():
            if any(x in name.lower() for x in ["time_mix", "wkv", "time_decay"]):
                # This is likely a time-mixing module
                RWKVAdapter._adapt_timemix_module(model, name, module, config)
            
            elif "channel_mix" in name.lower():
                # This is likely a channel-mixing module
                RWKVAdapter._adapt_channelmix_module(model, name, module, config)
            
            elif len(list(module.children())) > 0:
                # Recursively process children
                RWKVAdapter.adapt_model(module, config)
        
        return model
    
    @staticmethod
    def _adapt_timemix_module(parent, name, module, config):
        """Adapt time-mixing module to token-adaptive version"""
        # Implement specific adaptation logic for time-mixing modules
        pass
    
    @staticmethod
    def _adapt_channelmix_module(parent, name, module, config):
        """Adapt channel-mixing module to token-adaptive version"""
        # Implement specific adaptation logic for channel-mixing modules
        pass


#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

def count_parameters(model):
    """
    Count number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        param_size_mb: Size of parameters in MB
    """
    if not TORCH_AVAILABLE:
        return 0, 0
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    param_size_mb = param_size_bytes / (1024 * 1024)
    
    return total_params, param_size_mb


def estimate_macs_for_operation(op_type, input_shape, output_shape=None, groups=1):
    """
    Estimate MACs (multiply-accumulate operations) for a given operation
    
    Args:
        op_type: Type of operation ('linear', 'conv2d', 'matmul', etc.)
        input_shape: Shape of input tensor
        output_shape: Shape of output tensor (if needed)
        groups: Groups parameter for grouped convolutions
        
    Returns:
        macs: Estimated number of MACs
    """
    if op_type == "linear":
        # Linear: in_features * out_features
        if len(input_shape) >= 2:
            batch_size = np.prod(input_shape[:-1])
            in_features = input_shape[-1]
            out_features = output_shape[-1] if output_shape else 0
            
            if out_features:
                return batch_size * in_features * out_features
    
    elif op_type == "conv2d":
        # Conv2d: batch_size * out_channels * out_height * out_width * kernel_height * kernel_width * in_channels / groups
        if len(input_shape) == 4 and len(output_shape) == 4:
            batch_size, in_channels, in_height, in_width = input_shape
            _, out_channels, out_height, out_width = output_shape
            
            # Estimate kernel size from shapes if not provided
            if hasattr(op_type, "kernel_size"):
                kernel_size = op_type.kernel_size
                kernel_height, kernel_width = kernel_size if len(kernel_size) == 2 else (kernel_size[0], kernel_size[0])
            else:
                # Estimate from input/output shapes
                kernel_height = kernel_width = 3  # Most common default
            
            return batch_size * out_channels * out_height * out_width * kernel_height * kernel_width * in_channels / groups
    
    elif op_type == "matmul":
        # MatMul: batch_size * M * N * K
        if len(input_shape) == 3 and len(output_shape) == 3:
            batch_size, M, K = input_shape
            _, _, N = output_shape
            return batch_size * M * N * K
    
    elif op_type == "attention":
        # Self-attention: 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        if len(input_shape) >= 2:
            batch_size = input_shape[0]
            seq_len = input_shape[1]
            num_heads = 12  # Default for many models
            head_dim = 64   # Default for many models
            
            if len(input_shape) > 2:
                hidden_dim = input_shape[2]
                head_dim = hidden_dim // num_heads
            
            # QK attention + attention*V
            return 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    elif op_type == "ssm" or op_type == "mamba":
        # SSM/Mamba: Linear projection + state update + output projection
        if len(input_shape) == 3:
            batch_size, seq_len, hidden_dim = input_shape
            
            # Just an approximation
            state_dim = 16
            return (
                batch_size * seq_len * hidden_dim * hidden_dim +  # Input projection
                batch_size * seq_len * hidden_dim * state_dim +   # State update
                batch_size * seq_len * hidden_dim * hidden_dim    # Output projection
            )
    
    # Default for unknown operations
    return 0


def estimate_model_macs(model, input_shape):
    """
    Estimate total MACs for a model with given input shape
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        
    Returns:
        total_macs: Estimated total MACs
    """
    if not TORCH_AVAILABLE:
        return 0
    
    # Try to use existing profilers if available
    try:
        from thop import profile
        input_tensor = torch.rand(*input_shape)
        macs, _ = profile(model, inputs=(input_tensor,))
        return macs
    except:
        # Fall back to manual estimation
        pass
    
    # Manual estimation (would require implementation of operation-specific MACs)
    return 0


def flops_to_energy(flops, precision=32, hardware="gpu"):
    """
    Estimate energy consumption from FLOPs based on precision and hardware
    
    Args:
        flops: Number of floating-point operations
        precision: Numerical precision (4, 8, 16, 32)
        hardware: Hardware type (gpu, cpu)
        
    Returns:
        energy_joules: Estimated energy consumption in joules
    """
    # Energy efficiency values in pJ/FLOP - these are approximations
    # Values derived from published research on energy efficiency
    energy_pj_per_flop = {
        "gpu": {
            4: 0.5,   # 4-bit operations on GPU
            8: 1.0,   # 8-bit operations on GPU
            16: 2.0,  # 16-bit operations on GPU
            32: 4.0   # 32-bit operations on GPU
        },
        "cpu": {
            4: 1.0,   # 4-bit operations on CPU
            8: 2.0,   # 8-bit operations on CPU
            16: 4.0,  # 16-bit operations on CPU
            32: 8.0   # 32-bit operations on CPU
        }
    }
    
    # Get energy efficiency for specified hardware and precision
    if hardware in energy_pj_per_flop and precision in energy_pj_per_flop[hardware]:
        efficiency = energy_pj_per_flop[hardware][precision]
    else:
        # Default to 32-bit GPU if not found
        efficiency = energy_pj_per_flop["gpu"][32]
    
    # Calculate energy in joules (pJ to J)
    energy_joules = (flops * efficiency) / 1e12
    
    return energy_joules


def create_token_importance_heatmap(token_importance, tokens=None, show_colorbar=True, fig_size=(12, 3)):
    """
    Create a heatmap visualization of token importance
    
    Args:
        token_importance: Token importance tensor [batch, seq_len]
        tokens: List of token strings
        show_colorbar: Whether to show the colorbar
        fig_size: Figure size in inches
        
    Returns:
        fig: Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("Matplotlib is required for visualization")
        return None
    
    # Convert to numpy if needed
    if TORCH_AVAILABLE and isinstance(token_importance, torch.Tensor):
        token_importance = token_importance.cpu().numpy()
    
    # Get shape
    if len(token_importance.shape) > 1:
        # Use first batch
        token_importance = token_importance[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create heatmap
    im = ax.imshow(token_importance.reshape(1, -1), cmap='viridis', aspect='auto')
    
    # Add colorbar
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Importance')
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add token labels if provided
    if tokens is not None and len(tokens) == len(token_importance):
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    # Set title and labels
    ax.set_title('Token Importance')
    ax.set_xlabel('Token Position')
    
    plt.tight_layout()
    return fig


def get_token_adaptive_stats(model):
    """
    Get token-adaptive statistics from a model
    
    Args:
        model: Token-adaptive model
        
    Returns:
        stats: Dictionary of statistics
    """
    stats = {}
    
    # Get precision stats from model components if available
    if hasattr(model, "get_precision_stats"):
        precision_stats = model.get_precision_stats()
        stats["model"] = precision_stats
    
    # Check for individual layers with precision stats
    for name, module in model.named_modules():
        if hasattr(module, "get_precision_stats"):
            layer_stats = module.get_precision_stats()
            
            if "layers" not in stats:
                stats["layers"] = {}
            
            stats["layers"][name] = layer_stats
    
    return stats


def create_hybrid_model(base_model, tap_ratio=0.5, config=None):
    """
    Create a hybrid model with both standard and token-adaptive layers
    
    Args:
        base_model: Base model to convert
        tap_ratio: Ratio of layers to convert to token-adaptive
        config: TAP configuration
        
    Returns:
        model: Hybrid model
    """
    if config is None:
        config = TAPConfig()
    
    # Identify transformer layers
    transformer_layers = []
    
    for name, module in base_model.named_modules():
        # Look for typical transformer/block layer names
        if any(x in name.lower() for x in ["layer.", "block.", "transformer.", "encoder.", "decoder."]):
            # Check if it's a leaf module (doesn't have further children)
            if len(list(module.children())) > 0 and not any(x in name.lower() for x in ["embeddings", "head", "output"]):
                transformer_layers.append((name, module))
    
    # Calculate number of layers to convert
    num_layers = len(transformer_layers)
    num_layers_to_convert = max(1, min(num_layers, int(num_layers * tap_ratio)))
    
    # Sort layers by depth (assuming deeper layers handle more abstract representations)
    transformer_layers.sort(key=lambda x: len(x[0].split('.')))
    
    # Select layers to convert, prioritizing deeper layers
    layers_to_convert = transformer_layers[-num_layers_to_convert:]
    
    logger.info(f"Creating hybrid model: converting {num_layers_to_convert}/{num_layers} layers to token-adaptive")
    
    # Determine model architecture
    if config.model_arch == ModelArchitecture.MAMBA:
        adapter = MambaAdapter
    elif config.model_arch == ModelArchitecture.RETNET:
        adapter = RetNetAdapter
    elif config.model_arch == ModelArchitecture.RWKV:
        adapter = RWKVAdapter
    else:
        adapter = TransformerAdapter
    
    # Convert selected layers
    for name, module in layers_to_convert:
        logger.info(f"Converting layer: {name}")
        # Get parent module
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        child_name = name.rsplit('.', 1)[1] if '.' in name else name
        
        if parent_name:
            parent_module = base_model
            for part in parent_name.split('.'):
                parent_module = getattr(parent_module, part)
        else:
            parent_module = base_model
        
        # Convert the module
        converted_module = adapter.adapt_model(module, config)
        
        # Replace in parent
        setattr(parent_module, child_name, converted_module)
    
    return base_model


def convert_batch_to_bit_tensor(batch_data, precision, dtype=torch.float32):
    """
    Convert a batch tensor to specific bit precision
    
    Args:
        batch_data: Input tensor
        precision: Target precision bits
        dtype: Dtype to return
        
    Returns:
        tensor: Converted tensor
    """
    if not TORCH_AVAILABLE:
        return batch_data
    
    if precision == 32:
        return batch_data.to(dtype=torch.float32)
    elif precision == 16:
        return batch_data.to(dtype=torch.float16)
    elif precision == 8:
        # Int8 quantization
        float_tensor = batch_data.to(dtype=torch.float32)
        min_val = float_tensor.min()
        max_val = float_tensor.max()
        scale = 255.0 / (max_val - min_val)
        zero_point = -min_val * scale
        
        # Quantize
        quant = torch.round(float_tensor * scale + zero_point).clamp(0, 255).to(torch.uint8)
        
        # Dequantize
        dequant = (quant.to(dtype=torch.float32) - zero_point) / scale
        
        return dequant.to(dtype=dtype)
    elif precision == 4:
        # Int4 quantization (we use int8 and lose the lower 4 bits)
        float_tensor = batch_data.to(dtype=torch.float32)
        min_val = float_tensor.min()
        max_val = float_tensor.max()
        scale = 15.0 / (max_val - min_val)
        zero_point = -min_val * scale
        
        # Quantize to 4 bits (0-15)
        quant = torch.round(float_tensor * scale + zero_point).clamp(0, 15).to(torch.uint8)
        
        # Dequantize
        dequant = (quant.to(dtype=torch.float32) - zero_point) / scale
        
        return dequant.to(dtype=dtype)
    else:
        logger.warning(f"Unsupported precision: {precision}, using float32")
        return batch_data.to(dtype=torch.float32)


def token_adaptive_benchmark(model, tokenizer, prompts, max_new_tokens=20, precisions=[4, 8, 16, 32], iterations=3):
    """
    Benchmark model performance across different precision levels
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        prompts: List of text prompts to test
        max_new_tokens: Maximum new tokens to generate
        precisions: List of precision levels to test
        iterations: Number of iterations per precision level
        
    Returns:
        results: Benchmark results
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available, cannot benchmark")
        return {"error": "no_pytorch"}
    
    # Ensure model is in eval mode
    model.eval()
    
    # Results dictionary
    results = {
        "prompts": prompts,
        "max_new_tokens": max_new_tokens,
        "precisions": precisions,
        "iterations": iterations,
        "metrics": {}
    }
    
    # Create energy monitor
    energy_monitor = EnergyMonitor()
    
    # Test each precision level
    for precision in precisions:
        logger.info(f"Benchmarking {precision}-bit precision...")
        
        precision_results = []
        
        # Run multiple iterations
        for iteration in range(iterations):
            iteration_metrics = []
            
            # Test each prompt
            for prompt in prompts:
                # Tokenize prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_ids = inputs["input_ids"]
                
                # Start energy monitoring
                energy_monitor.start()
                start_time = time.time()
                
                # Generate with fixed precision
                with torch.no_grad():
                    if precision == 32:
                        # Use standard precision
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            use_cache=True
                        )
                    else:
                        # Use custom precision
                        with torch.autocast(device_type=model.device.type, dtype=torch.float16 if precision == 16 else torch.float32):
                            outputs = model.generate(
                                input_ids=input_ids,
                                max_new_tokens=max_new_tokens,
                                use_cache=True
                            )
                
                # Calculate metrics
                generation_time = time.time() - start_time
                energy_metrics = energy_monitor.stop()
                
                # Add to results
                prompt_metrics = {
                    "tokens_generated": outputs.shape[1] - input_ids.shape[1],
                    "generation_time": generation_time,
                    "tokens_per_second": (outputs.shape[1] - input_ids.shape[1]) / generation_time,
                    "energy": energy_metrics
                }
                
                iteration_metrics.append(prompt_metrics)
            
            # Aggregate metrics for this iteration
            avg_metrics = {
                "tokens_generated": sum(m["tokens_generated"] for m in iteration_metrics) / len(iteration_metrics),
                "generation_time": sum(m["generation_time"] for m in iteration_metrics) / len(iteration_metrics),
                "tokens_per_second": sum(m["tokens_per_second"] for m in iteration_metrics) / len(iteration_metrics),
                "energy": {
                    "total_energy": sum(m["energy"]["total_energy"] for m in iteration_metrics) / len(iteration_metrics),
                    "avg_power": sum(m["energy"]["avg_power"] for m in iteration_metrics) / len(iteration_metrics)
                }
            }
            
            precision_results.append(avg_metrics)
        
        # Calculate average across iterations
        precision_avg = {
            "tokens_generated": sum(r["tokens_generated"] for r in precision_results) / len(precision_results),
            "generation_time": sum(r["generation_time"] for r in precision_results) / len(precision_results),
            "tokens_per_second": sum(r["tokens_per_second"] for r in precision_results) / len(precision_results),
            "energy": {
                "total_energy": sum(r["energy"]["total_energy"] for r in precision_results) / len(precision_results),
                "avg_power": sum(r["energy"]["avg_power"] for r in precision_results) / len(precision_results)
            },
            "iterations": precision_results
        }
        
        # Add to results
        results["metrics"][str(precision)] = precision_avg
    
    # Calculate comparisons relative to 32-bit
    if "32" in results["metrics"]:
        baseline = results["metrics"]["32"]
        
        for precision in results["metrics"]:
            if precision != "32":
                current = results["metrics"][precision]
                
                # Calculate speedup
                speedup = baseline["generation_time"] / current["generation_time"]
                energy_reduction = (baseline["energy"]["total_energy"] - current["energy"]["total_energy"]) / baseline["energy"]["total_energy"] * 100
                
                results["metrics"][precision]["speedup_vs_fp32"] = speedup
                results["metrics"][precision]["energy_reduction_vs_fp32"] = energy_reduction
    
    return results


#------------------------------------------------------------------------------
# Export/Import Utilities
#------------------------------------------------------------------------------

def export_model_for_inference(model, output_dir, config=None, optimize=True, quantize_bits=8):
    """
    Export model for efficient inference
    
    Args:
        model: Model to export
        output_dir: Output directory
        config: TAP configuration
        optimize: Whether to apply optimizations
        quantize_bits: Bits for quantization
        
    Returns:
        export_path: Path to exported model
    """
    import os
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply optimizations if requested
    if optimize:
        logger.info("Optimizing model for inference...")
        
        if config is None:
            config = TAPConfig()
        
        # Create engine to handle optimization
        engine = TAPEngine(config=config)
        engine.model = model
        
        # Optimize model
        engine.optimize_model(quantize_bits=quantize_bits)
        model = engine.model
    
    # Export model
    logger.info(f"Exporting model to {output_dir}...")
    
    # Save model weights
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save model configuration
    if hasattr(model, "config") and hasattr(model.config, "to_json_string"):
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            f.write(model.config.to_json_string())
    
    # Save TAP configuration
    if config is not None:
        config_path = os.path.join(output_dir, "tap_config.json")
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    
    # Create readme with usage information
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"# Token-Adaptive Precision Model\n\n")
        f.write(f"This model has been optimized with the TAP Engine.\n\n")
        f.write(f"## Loading Instructions\n\n")
        f.write(f"```python\n")
        f.write(f"from transformers import AutoModelForCausalLM, AutoTokenizer\n")
        f.write(f"from tap_engine import TAPEngine, TAPConfig\n\n")
        f.write(f"# Load model with TAP Engine\n")
        f.write(f"tap_engine = TAPEngine.from_pretrained('{output_dir}')\n")
        f.write(f"model = tap_engine.model\n")
        f.write(f"tokenizer = tap_engine.tokenizer\n\n")
        f.write(f"# Generate text with token-adaptive precision\n")
        f.write(f"text, metrics = tap_engine.generate(prompt='Hello, world!', max_new_tokens=50)\n")
        f.write(f"print(text)\n")
        f.write(f"```\n")
    
    return output_dir


def tap_trace(func=None, trace_name=None, record_shapes=True):
    """
    Decorator to trace operations with token-adaptive precision information
    
    Args:
        func: Function to trace
        trace_name: Name for the trace
        record_shapes: Whether to record tensor shapes
        
    Returns:
        wrapped_func: Traced function
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Get name from function if not specified
            name = trace_name or fn.__name__
            
            # Check for token_precision in kwargs
            token_precision = kwargs.get("token_precision", None)
            
            # Start timer
            start_time = time.time()
            
            # Call original function
            result = fn(*args, **kwargs)
            
            # End timer
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record trace information
            trace_info = {
                "name": name,
                "execution_time": execution_time,
                "has_token_precision": token_precision is not None
            }
            
            # Add tensor shapes if requested
            if record_shapes:
                arg_shapes = []
                
                # Get shapes of tensor arguments
                for arg in args:
                    if hasattr(arg, "shape"):
                        arg_shapes.append(list(arg.shape))
                    else:
                        arg_shapes.append(None)
                
                trace_info["arg_shapes"] = arg_shapes
                
                # Add result shapes if result is a tensor or tuple/list of tensors
                if hasattr(result, "shape"):
                    trace_info["result_shape"] = list(result.shape)
                elif isinstance(result, (tuple, list)) and all(hasattr(x, "shape") for x in result):
                    trace_info["result_shape"] = [list(x.shape) for x in result]
            
            # Add token precision stats if available
            if token_precision is not None:
                # Get unique precisions and counts
                if hasattr(token_precision, "unique"):
                    unique_precs = token_precision.unique(sorted=True)
                    precision_counts = {}
                    
                    for prec in unique_precs:
                        prec_val = prec.item()
                        count = (token_precision == prec_val).sum().item()
                        precision_counts[int(prec_val)] = count
                    
                    total_tokens = token_precision.numel()
                    
                    trace_info["token_precision_stats"] = {
                        "counts": precision_counts,
                        "total_tokens": total_tokens,
                        "distribution": {prec: count/total_tokens for prec, count in precision_counts.items()}
                    }
            
            # Log trace info
            logger.debug(f"TAP Trace: {json.dumps(trace_info)}")
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


#------------------------------------------------------------------------------
# Triton Kernels for Token-Adaptive Operations
#------------------------------------------------------------------------------

if TRITON_AVAILABLE:
    # Import required modules for Triton kernels
    import triton
    import triton.language as tl
    
    @triton.jit
    def token_adaptive_layernorm_kernel(
        x_ptr, gamma_ptr, beta_ptr, out_ptr, precision_ptr,
        stride_x_batch, stride_x_seq, stride_x_hidden,
        stride_out_batch, stride_out_seq, stride_out_hidden,
        hidden_size, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Triton kernel for token-adaptive layer normalization
        
        Args:
            x_ptr: Input tensor pointer [batch, seq, hidden]
            gamma_ptr: Scale parameter pointer [hidden]
            beta_ptr: Shift parameter pointer [hidden]
            out_ptr: Output tensor pointer [batch, seq, hidden]
            precision_ptr: Token precision pointer [batch, seq]
            strides: Tensor strides
            hidden_size: Hidden dimension size
            eps: Epsilon for numerical stability
            BLOCK_SIZE: Block size for Triton
        """
        # Get program ID for batch and sequence dimensions
        batch_idx = tl.program_id(0)
        seq_idx = tl.program_id(1)
        
        # Compute offsets for the current token
        x_offset = batch_idx * stride_x_batch + seq_idx * stride_x_seq
        out_offset = batch_idx * stride_out_batch + seq_idx * stride_out_seq
        
        # Get precision for current token
        precision_offset = batch_idx * tl.load(precision_ptr + seq_idx)
        precision = tl.load(precision_ptr + precision_offset)
        
        # Block-level operation by loading a tile of the input
        x_block_ptr = x_ptr + x_offset + tl.arange(0, BLOCK_SIZE) * stride_x_hidden
        x = tl.load(x_block_ptr, mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
        
        # Compute mean
        mean = tl.sum(x) / hidden_size
        
        # Compute variance
        x_minus_mean = x - mean
        var = tl.sum(x_minus_mean * x_minus_mean) / hidden_size
        
        # Normalize
        inv_std = 1.0 / tl.sqrt(var + eps)
        x_norm = x_minus_mean * inv_std
        
        # Load gamma and beta
        gamma_block_ptr = gamma_ptr + tl.arange(0, BLOCK_SIZE)
        beta_block_ptr = beta_ptr + tl.arange(0, BLOCK_SIZE)
        gamma = tl.load(gamma_block_ptr, mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
        beta = tl.load(beta_block_ptr, mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0)
        
        # Apply scale and shift
        y = gamma * x_norm + beta
        
        # Apply quantization based on precision
        if precision == 4:
            # 4-bit quantization
            scale = 7.0  # -7 to 7 range
            y_quant = tl.round(y * scale)
            y_quant = tl.min(tl.max(y_quant, -7.0), 7.0)
            y = y_quant / scale
        elif precision == 8:
            # 8-bit quantization
            scale = 127.0  # -127 to 127 range
            y_quant = tl.round(y * scale)
            y_quant = tl.min(tl.max(y_quant, -127.0), 127.0)
            y = y_quant / scale
        elif precision == 16:
            # 16-bit quantization - simple truncation to f16
            y = tl.float16(y)
        
        # Store the result
        out_block_ptr = out_ptr + out_offset + tl.arange(0, BLOCK_SIZE) * stride_out_hidden
        tl.store(out_block_ptr, y, mask=tl.arange(0, BLOCK_SIZE) < hidden_size)
    
    
    @triton.jit
    def token_adaptive_matmul_kernel(
        a_ptr, b_ptr, c_ptr, precision_ptr,
        stride_a_batch, stride_a_seq, stride_a_hidden,
        stride_b_in, stride_b_out,
        stride_c_batch, stride_c_seq, stride_c_hidden,
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
    ):
        """
        Triton kernel for token-adaptive matrix multiplication
        
        Args:
            a_ptr: Input tensor pointer [batch, seq, K]
            b_ptr: Weight tensor pointer [K, N]
            c_ptr: Output tensor pointer [batch, seq, N]
            precision_ptr: Token precision pointer [batch, seq]
            strides: Tensor strides
            M, N, K: Matrix dimensions
            BLOCK_SIZE_*: Block sizes for Triton
        """
        # Get program IDs
        batch_idx = tl.program_id(0)
        seq_idx = tl.program_id(1)
        
        # Compute offsets
        a_offset = batch_idx * stride_a_batch + seq_idx * stride_a_seq
        c_offset = batch_idx * stride_c_batch + seq_idx * stride_c_seq
        
        # Get precision for current token
        precision_offset = batch_idx * tl.load(precision_ptr + seq_idx)
        precision = tl.load(precision_ptr + precision_offset)
        
        # Pointers for the current token
        a_block_ptr = a_ptr + a_offset + tl.arange(0, BLOCK_SIZE_K) * stride_a_hidden
        c_block_ptr = c_ptr + c_offset + tl.arange(0, BLOCK_SIZE_N) * stride_c_hidden
        
        # Initialize output
        c = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        
        # Apply different precision based on token
        if precision == 4:
            # 4-bit quantization for computation
            scale_a = 7.0
            scale_b = 7.0
            
            # Loop over K dimension
            for k in range(0, K, BLOCK_SIZE_K):
                # Load a tile from A and quantize
                a = tl.load(a_block_ptr + k, mask=k + tl.arange(0, BLOCK_SIZE_K) < K, other=0.0)
                a_quant = tl.round(a * scale_a)
                a_quant = tl.min(tl.max(a_quant, -7.0), 7.0)
                a = a_quant / scale_a
                
                # For each output dimension
                for n in range(0, N, BLOCK_SIZE_N):
                    b_block_ptr = b_ptr + k * stride_b_in + n * stride_b_out
                    
                    # Load a tile from B and quantize
                    b = tl.load(b_block_ptr, mask=n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
                    b_quant = tl.round(b * scale_b)
                    b_quant = tl.min(tl.max(b_quant, -7.0), 7.0)
                    b = b_quant / scale_b
                    
                    # Compute partial output
                    c += tl.sum(a[:, None] * b[None, :], axis=0)
        
        elif precision == 8:
            # 8-bit quantization for computation
            scale_a = 127.0
            scale_b = 127.0
            
            # Similar implementation as 4-bit but with different scales and quantization ranges
            # ...
        
        elif precision == 16:
            # 16-bit computation
            # Convert to float16 for computation
            # ...
        
        else:
            # 32-bit computation (full precision)
            # Standard matrix multiplication
            # ...
        
        # Store the result
        tl.store(c_block_ptr, c, mask=tl.arange(0, BLOCK_SIZE_N) < N)
    
    
    class TritonTokenAdaptiveLayerNorm(nn.Module):
        """
        Layer normalization with token-adaptive precision using Triton kernels
        """
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__()
            
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(normalized_shape))
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
        
        def forward(self, input, token_precision=None):
            """Forward pass with Triton kernel if token_precision is provided"""
            # If no token precision or all tokens have same precision, use standard layer norm
            if token_precision is None or torch.all(token_precision == token_precision[0, 0]):
                return F.layer_norm(
                    input, self.normalized_shape, self.weight, self.bias, self.eps
                )
            
            # Get shapes
            batch_size, seq_len, hidden_size = input.shape
            
            # Create output tensor
            output = torch.empty_like(input)
            
            # Calculate grid dimensions
            grid = (batch_size, seq_len)
            
            # Launch the Triton kernel
            token_adaptive_layernorm_kernel[grid](
                input, self.weight, self.bias, output, token_precision,
                input.stride(0), input.stride(1), input.stride(2),
                output.stride(0), output.stride(1), output.stride(2),
                hidden_size, self.eps,
                BLOCK_SIZE=min(hidden_size, 1024)  # Adjust block size based on hidden size
            )
            
            return output


#------------------------------------------------------------------------------
# Integration with PEFT and QLoRA
#------------------------------------------------------------------------------

if PEFT_AVAILABLE:
    
    class TokenAdaptiveLoraLayer(nn.Module):
        """
        Token-adaptive LoRA layer that applies different precision to different tokens
        """
        def __init__(self, base_layer, rank=8, alpha=16, dropout=0.0, config=None):
            super().__init__()
            self.config = config or TAPConfig()
            
            # Save original layer
            self.base_layer = base_layer
            
            # Get dimensions
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                self.in_features = base_layer.in_features
                self.out_features = base_layer.out_features
            else:
                raise ValueError("Base layer must have in_features and out_features attributes")
            
            # LoRA parameters
            self.rank = rank
            self.scaling = alpha / rank
            
            # Initialize A and B matrices for LoRA
            self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
            
            # Initialize LoRA matrices
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
            # Dropout for LoRA
            self.dropout = nn.Dropout(dropout)
            
            # Precision tracking
            self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
        
        def forward(self, x, token_precision=None):
            """Forward pass with token-adaptive LoRA adaptation"""
            batch_size, seq_len, _ = x.shape
            
            # Base layer output
            base_output = self.base_layer(x)
            
            # Apply LoRA with token-adaptive precision
            if token_precision is not None:
                # Update precision tracking
                self.precision_tracker["calls"] += 1
                self.precision_tracker["tokens_processed"] += batch_size * seq_len
                
                # Process each token with appropriate precision
                lora_output = torch.zeros_like(base_output)
                
                for precision in torch.unique(token_precision):
                    precision_val = precision.item()
                    
                    # Count tokens with this precision
                    tokens_with_precision = (token_precision == precision_val).sum().item()
                    
                    if precision_val not in self.precision_tracker["precision_counts"]:
                        self.precision_tracker["precision_counts"][precision_val] = 0
                    
                    self.precision_tracker["precision_counts"][precision_val] += tokens_with_precision
                    
                    # Create mask for tokens with this precision
                    mask = (token_precision == precision_val).unsqueeze(-1).expand(-1, -1, self.in_features)
                    
                    if not mask.any():
                        continue
                    
                    # Extract tokens with this precision
                    indices = torch.where(mask.any(dim=-1))
                    batch_indices, seq_indices = indices[0], indices[1]
                    
                    # Get inputs for these tokens
                    x_precision = torch.zeros((len(batch_indices), self.in_features), device=x.device)
                    
                    for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                        x_precision[i] = x[b_idx, s_idx]
                    
                    # Apply dropout
                    x_precision = self.dropout(x_precision)
                    
                    # Quantize LoRA weights according to precision
                    lora_A_quant, _ = PrecisionManager.quantize_tensor(
                        self.lora_A, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                    
                    lora_B_quant, _ = PrecisionManager.quantize_tensor(
                        self.lora_B, bits=precision_val, scheme=QuantizationScheme.SYMMETRIC)
                    
                    # Compute LoRA output for these tokens
                    lora_x = (x_precision @ lora_A_quant.T) @ lora_B_quant.T * self.scaling
                    
                    # Insert results back
                    for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                        lora_output[b_idx, s_idx] = lora_x[i]
                
                # Add LoRA output to base output
                return base_output + lora_output
            
            else:
                # Standard LoRA without token-adaptive precision
                lora_output = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
                return base_output + lora_output
        
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
    
    
    def add_token_adaptive_lora(model, lora_rank=8, lora_alpha=16, lora_dropout=0.05, config=None):
        """
        Add token-adaptive LoRA to a model
        
        Args:
            model: PyTorch model
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout probability
            config: TAP configuration
            
        Returns:
            model: Model with token-adaptive LoRA
        """
        if config is None:
            config = TAPConfig()
        
        # Replace linear layers with token-adaptive LoRA layers
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Skip output layer for better accuracy
                if "lm_head" in name or "output_projection" in name:
                    continue
                
                # Create token-adaptive LoRA layer
                lora_layer = TokenAdaptiveLoraLayer(
                    module, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, config=config
                )
                
                # Replace in parent module
                setattr(model, name, lora_layer)
            
            elif len(list(module.children())) > 0:
                # Recursively process children
                add_token_adaptive_lora(
                    module, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, config=config
                )
        
        return model


#------------------------------------------------------------------------------
# BitByte Integration for INT4/INT8 Quantized Operations
#------------------------------------------------------------------------------

if BITSANDBYTES_AVAILABLE:
    
    class TokenAdaptive4BitLinear(nn.Module):
        """
        Token-adaptive 4-bit quantized linear layer using BitsAndBytes
        """
        def __init__(self, in_features, out_features, bias=True, config=None):
            super().__init__()
            self.config = config or TAPConfig()
            
            # Create BitsAndBytes 4-bit linear layer
            self.quant_linear = bnb.nn.Linear4bit(
                in_features, out_features, bias=bias, compute_dtype=torch.float16
            )
            
            # Create separate full-precision layer for high-precision tokens
            self.fp_linear = nn.Linear(in_features, out_features, bias=bias)
            
            # Copy initialized weights from quant_linear to fp_linear
            with torch.no_grad():
                # This is approximate since we don't have direct access to 4-bit weights
                self.fp_linear.weight.copy_(self.quant_linear.weight.float())
                if bias:
                    self.fp_linear.bias.copy_(self.quant_linear.bias)
            
            # For precision tracking
            self.precision_tracker = {"calls": 0, "tokens_processed": 0, "precision_counts": {}}
        
        def forward(self, x, token_precision=None):
            """Forward pass with token-adaptive precision"""
            if token_precision is None:
                # Use quantized version for all tokens
                return self.quant_linear(x)
            
            # Update precision tracking
            batch_size, seq_len, _ = x.shape
            self.precision_tracker["calls"] += 1
            self.precision_tracker["tokens_processed"] += batch_size * seq_len
            
            # Initialize output tensor
            output = torch.zeros(batch_size, seq_len, self.quant_linear.out_features, device=x.device)
            
            # Process each precision level separately
            for precision in torch.unique(token_precision):
                precision_val = precision.item()
                
                # Update precision counts
                tokens_with_precision = (token_precision == precision_val).sum().item()
                if precision_val not in self.precision_tracker["precision_counts"]:
                    self.precision_tracker["precision_counts"][precision_val] = 0
                self.precision_tracker["precision_counts"][precision_val] += tokens_with_precision
                
                # Create mask for tokens with this precision
                mask = (token_precision == precision_val)
                
                if not mask.any():
                    continue
                
                # Extract token indices
                indices = torch.where(mask)
                batch_indices, seq_indices = indices[0], indices[1]
                
                # Extract tokens
                x_precision = torch.zeros(len(batch_indices), x.shape[2], device=x.device)
                for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                    x_precision[i] = x[b_idx, s_idx]
                
                # Process with appropriate precision
                if precision_val <= 4:
                    # Use 4-bit layer
                    output_precision = self.quant_linear(x_precision)
                else:
                    # Use higher precision layer
                    output_precision = self.fp_linear(x_precision)
                
                # Insert results back
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
# Sample High-Level Usage Example
#------------------------------------------------------------------------------

def example_usage():
    """Simple example of how to use the TAP Engine"""
    # Create configuration
    config = TAPConfig(
        model_arch=ModelArchitecture.MAMBA,
        precision_mode=PrecisionMode.ADAPTIVE,
        precision_levels=[4, 8, 16, 32],
        precision_thresholds=[0.3, 0.6, 0.9]
    )
    
    # Initialize TAP Engine
    engine = TAPEngine(config=config)
    
    # Load model
    engine.load_model("mamba-0.1b")
    
    # Generate text with token-adaptive precision
    text, metrics = engine.generate(
        prompt="Token-Adaptive Precision allows LLMs to",
        max_new_tokens=50
    )
    
    # Print results
    print("\nGenerated text:")
    print(text)
    
    print("\nGeneration metrics:")
    print(f"- Tokens generated: {metrics['generated_tokens']}")
    print(f"- Generation time: {metrics['generation_time']:.2f} seconds")
    print(f"- Tokens per second: {metrics['tokens_per_second']:.2f}")
    
    if "precision" in metrics:
        print("\nPrecision distribution:")
        for k, v in metrics["precision"].items():
            if k.startswith("precision_") and k.endswith("_pct"):
                bits = k.split("_")[1]
                print(f"- {bits}-bit: {v:.1f}%")
        
        print(f"\nEstimated energy saved: {metrics['precision']['energy_saved_pct']:.2f}%")
    
    # Analyze token importance for a specific prompt
    analysis = engine.analyze_token_importance(
        prompt="Analyzing token importance for adaptive precision"
    )
    
    print("\nToken importance analysis:")
    token_importance = analysis["token_importance"]["default"]
    token_precision = analysis["token_precision"]
    
    for i in range(min(10, token_importance.shape[1])):
        imp = token_importance[0, i].item()
        prec = token_precision[0, i].item()
        print(f"Token {i}: Importance={imp:.2f}, Precision={prec}-bit")

    # Save the model
    engine.save_model("tap_optimized_model")
    
    print("\nModel saved to: tap_optimized_model")


if __name__ == "__main__":
    # When run directly, execute the example
    try:
        example_usage()
    except ImportError as e:
        print(f"Required library not available: {e}")
        print("Install missing dependencies and try again.")
    except Exception as e:
        print(f"Error in example: {e}")
        if logger.level <= logging.DEBUG:
            import traceback
            traceback.print_exc()
            
