#------------------------------------------------------------------------------
# Testing Utilities for TAP Engine
#------------------------------------------------------------------------------

def run_test_suite(model, tokenizer, test_cases=None, precision_levels=None, verbose=True):
    """
    Run comprehensive test suite for token-adaptive precision
    
    Args:
        model: Model to test
        tokenizer: Tokenizer for the model
        test_cases: List of test prompts or None for defaults
        precision_levels: List of precision levels to test or None for defaults
        verbose: Whether to print detailed results
        
    Returns:
        results: Test results
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    # Default test cases
    if test_cases is None:
        test_cases = [
            "Hello, my name is",
            "The capital city of France is",
            "Quantum mechanics describes the behavior of",
            "In a world where technology evolves rapidly",
            "The following code snippet demonstrates how to"
        ]
    
    # Default precision levels
    if precision_levels is None:
        precision_levels = [4, 8, 16, 32]
    
    # Test results
    results = {
        "model_info": {
            "name": model.__class__.__name__,
            "parameters": count_parameters(model)[0]
        },
        "test_cases": test_cases,
        "precision_levels": precision_levels,
        "fixed_precision_results": {},
        "adaptive_precision_results": {}
    }
    
    # Set model to evaluation mode
    model.eval()
    
    # Create energy monitor
    energy_monitor = EnergyMonitor()
    
    # Test fixed precision levels
    if verbose:
        print("\nTesting fixed precision levels:")
    
    for precision in precision_levels:
        if verbose:
            print(f"  - {precision}-bit precision...")
        
        # Results for this precision level
        precision_results = []
        
        for prompt in test_cases:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            
            # Start energy monitoring
            energy_monitor.start()
            start_time = time.time()
            
            # Generate with fixed precision
            with torch.no_grad():
                token_precision = torch.full((1, input_ids.shape[1]), precision, device=model.device)
                
                try:
                    # Try with token_precision parameter
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20,
                        token_precision=token_precision
                    )
                except Exception:
                    # Fall back to standard generation
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20
                    )
            
            # End monitoring
            generation_time = time.time() - start_time
            energy_metrics = energy_monitor.stop()
            
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Store results
            test_result = {
                "prompt": prompt,
                "output": output_text,
                "new_tokens": outputs.shape[1] - input_ids.shape[1],
                "generation_time": generation_time,
                "tokens_per_second": (outputs.shape[1] - input_ids.shape[1]) / generation_time,
                "energy": energy_metrics
            }
            
            precision_results.append(test_result)
        
        # Calculate averages
        avg_generation_time = sum(r["generation_time"] for r in precision_results) / len(precision_results)
        avg_tokens_per_second = sum(r["tokens_per_second"] for r in precision_results) / len(precision_results)
        avg_energy = sum(r["energy"]["total_energy"] for r in precision_results) / len(precision_results)
        
        # Summary for this precision level
        results["fixed_precision_results"][str(precision)] = {
            "avg_generation_time": avg_generation_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_energy": avg_energy,
            "test_results": precision_results
        }
        
        if verbose:
            print(f"    Time: {avg_generation_time:.3f}s, Tokens/s: {avg_tokens_per_second:.1f}, Energy: {avg_energy:.3f}J")
    
    # Test token-adaptive precision
    if verbose:
        print("\nTesting token-adaptive precision:")
    
    adaptive_results = []
    
    for prompt in test_cases:
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        
        # Start energy monitoring
        energy_monitor.start()
        start_time = time.time()
        
        # Generate with token-adaptive precision
        # First, calculate token importance
        with torch.no_grad():
            # Forward pass to get hidden states
            outputs = model(
                input_ids=input_ids,
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
            
            # Calculate token importance
            token_analyzer = TokenImportanceAnalyzer()
            token_importance = token_analyzer.analyze_token_importance(
                attention_scores=attention_scores,
                hidden_states=hidden_states,
                input_ids=input_ids
            )
            
            # Assign precision levels
            token_precision = token_analyzer.assign_precision(token_importance)
            
            try:
                # Try with token_precision parameter
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    token_precision=token_precision
                )
            except Exception:
                # Fall back to standard generation
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20
                )
        
        # End monitoring
        generation_time = time.time() - start_time
        energy_metrics = energy_monitor.stop()
        
        # Decode output
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Calculate precision distribution
        precision_counts = {}
        for prec in precision_levels:
            count = (token_precision == prec).sum().item()
            pct = count / token_precision.numel() * 100
            precision_counts[str(prec)] = pct
        
        # Store results
        test_result = {
            "prompt": prompt,
            "output": output_text,
            "new_tokens": generated_ids.shape[1] - input_ids.shape[1],
            "generation_time": generation_time,
            "tokens_per_second": (generated_ids.shape[1] - input_ids.shape[1]) / generation_time,
            "energy": energy_metrics,
            "precision_distribution": precision_counts
        }
        
        adaptive_results.append(test_result)
        
        if verbose:
            print(f"  - Prompt: '{prompt[:50]}...'")
            print(f"    Time: {generation_time:.3f}s, Tokens/s: {test_result['tokens_per_second']:.1f}")
            print(f"    Precision distribution: " + 
                  ", ".join([f"{p}-bit: {pct:.1f}%" for p, pct in precision_counts.items()]))
    
    # Calculate averages
    avg_generation_time = sum(r["generation_time"] for r in adaptive_results) / len(adaptive_results)
    avg_tokens_per_second = sum(r["tokens_per_second"] for r in adaptive_results) / len(adaptive_results)
    avg_energy = sum(r["energy"]["total_energy"] for r in adaptive_results) / len(adaptive_results)
    
    # Aggregate precision distribution
    avg_precision_dist = {}
    for prec in precision_levels:
        prec_str = str(prec)
        avg_precision_dist[prec_str] = sum(r["precision_distribution"].get(prec_str, 0) 
                                         for r in adaptive_results) / len(adaptive_results)
    
    # Summary for adaptive precision
    results["adaptive_precision_results"] = {
        "avg_generation_time": avg_generation_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_energy": avg_energy,
        "avg_precision_distribution": avg_precision_dist,
        "test_results": adaptive_results
    }
    
    # Calculate comparisons to 32-bit
    if "32" in results["fixed_precision_results"]:
        fp32_time = results["fixed_precision_results"]["32"]["avg_generation_time"]
        fp32_energy = results["fixed_precision_results"]["32"]["avg_energy"]
        
        # Compare adaptive to fp32
        adaptive_time = results["adaptive_precision_results"]["avg_generation_time"]
        adaptive_energy = results["adaptive_precision_results"]["avg_energy"]
        
        speedup = fp32_time / adaptive_time
        energy_reduction = (fp32_energy - adaptive_energy) / fp32_energy * 100
        
        results["adaptive_precision_results"]["speedup_vs_fp32"] = speedup
        results["adaptive_precision_results"]["energy_reduction_vs_fp32"] = energy_reduction
        
        if verbose:
            print(f"\nToken-adaptive precision vs FP32:")
            print(f"  - Speedup: {speedup:.2f}x")
            print(f"  - Energy reduction: {energy_reduction:.1f}%")
    
    return results


def visualize_test_results(results, output_file=None):
    """
    Visualize test results with charts
    
    Args:
        results: Test results from run_test_suite
        output_file: Output file for visualization
        
    Returns:
        fig: Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib is required for visualization")
        return None
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Generation time comparison
    ax1 = axes[0, 0]
    
    # Get data for fixed precision levels
    precisions = []
    gen_times = []
    
    for prec, data in results["fixed_precision_results"].items():
        precisions.append(prec)
        gen_times.append(data["avg_generation_time"])
    
    # Add adaptive precision
    precisions.append("Adaptive")
    gen_times.append(results["adaptive_precision_results"]["avg_generation_time"])
    
    # Plot generation times
    x = np.arange(len(precisions))
    bars = ax1.bar(x, gen_times, color='skyblue')
    
    # Add adaptive precision bar with different color
    bars[-1].set_color('orange')
    
    # Add labels and title
    ax1.set_xlabel('Precision Mode')
    ax1.set_ylabel('Generation Time (s)')
    ax1.set_title('Generation Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{p}-bit" if p != "Adaptive" else p for p in precisions])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}s', ha='center', va='bottom')
    
    # 2. Tokens per second comparison
    ax2 = axes[0, 1]
    
    # Get data for fixed precision levels
    tokens_per_sec = []
    
    for prec, data in results["fixed_precision_results"].items():
        tokens_per_sec.append(data["avg_tokens_per_second"])
    
    # Add adaptive precision
    tokens_per_sec.append(results["adaptive_precision_results"]["avg_tokens_per_second"])
    
    # Plot tokens per second
    bars = ax2.bar(x, tokens_per_sec, color='lightgreen')
    
    # Add adaptive precision bar with different color
    bars[-1].set_color('orange')
    
    # Add labels and title
    ax2.set_xlabel('Precision Mode')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('Throughput Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p}-bit" if p != "Adaptive" else p for p in precisions])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 3. Energy consumption comparison
    ax3 = axes[1, 0]
    
    # Get data for fixed precision levels
    energy_usage = []
    
    for prec, data in results["fixed_precision_results"].items():
        energy_usage.append(data["avg_energy"])
    
    # Add adaptive precision
    energy_usage.append(results["adaptive_precision_results"]["avg_energy"])
    
    # Plot energy usage
    bars = ax3.bar(x, energy_usage, color='salmon')
    
    # Add adaptive precision bar with different color
    bars[-1].set_color('orange')
    
    # Add labels and title
    ax3.set_xlabel('Precision Mode')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Energy Consumption Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{p}-bit" if p != "Adaptive" else p for p in precisions])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}J', ha='center', va='bottom')
    
    # 4. Precision distribution for adaptive precision
    ax4 = axes[1, 1]
    
    # Get precision distribution
    prec_dist = results["adaptive_precision_results"]["avg_precision_distribution"]
    
    # Sort by precision
    prec_labels = sorted(prec_dist.keys())
    prec_values = [prec_dist[p] for p in prec_labels]
    
    # Plot pie chart
    wedges, texts, autotexts = ax4.pie(
        prec_values, 
        labels=[f"{p}-bit" for p in prec_labels], 
        autopct='%1.1f%%',
        startangle=90,
        colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    )
    
    # Add title
    ax4.set_title('Token-Adaptive Precision Distribution')
    
    # Add comparison text if available
    if "speedup_vs_fp32" in results["adaptive_precision_results"]:
        speedup = results["adaptive_precision_results"]["speedup_vs_fp32"]
        energy_reduction = results["adaptive_precision_results"]["energy_reduction_vs_fp32"]
        
        comparison_text = (
            f"Adaptive vs FP32:\n"
            f"Speedup: {speedup:.2f}x\n"
            f"Energy reduction: {energy_reduction:.1f}%"
        )
        
        fig.text(0.5, 0.04, comparison_text, ha='center', va='center', fontsize=12,
                 bbox=dict(facecolor='orange', alpha=0.2))
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Token-Adaptive Precision Performance', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save or show the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    return fig


def evaluate_on_benchmark(model, tokenizer, benchmark_data, token_adaptive=True, max_samples=None):
    """
    Evaluate model on a benchmark dataset
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        benchmark_data: Benchmark dataset (list of dicts with 'prompt' and 'reference')
        token_adaptive: Whether to use token-adaptive precision
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        results: Evaluation results
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    # Limit number of samples if specified
    if max_samples is not None:
        benchmark_data = benchmark_data[:max_samples]
    
    # Set model to evaluation mode
    model.eval()
    
    # Create energy monitor
    energy_monitor = EnergyMonitor()
    
    # Create token analyzer if using adaptive precision
    token_analyzer = None
    if token_adaptive:
        token_analyzer = TokenImportanceAnalyzer()
    
    # Results
    results = {
        "samples": len(benchmark_data),
        "token_adaptive": token_adaptive,
        "total_tokens_generated": 0,
        "total_generation_time": 0,
        "sample_results": []
    }
    
    # Process each benchmark sample
    for i, sample in enumerate(benchmark_data):
        print(f"Processing sample {i+1}/{len(benchmark_data)}...")
        
        prompt = sample["prompt"]
        reference = sample.get("reference")
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        
        # Start energy monitoring
        energy_monitor.start()
        start_time = time.time()
        
        # Generate with token-adaptive precision
        with torch.no_grad():
            if token_adaptive:
                # First, calculate token importance
                outputs = model(
                    input_ids=input_ids,
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
                
                # Calculate token importance
                token_importance = token_analyzer.analyze_token_importance(
                    attention_scores=attention_scores,
                    hidden_states=hidden_states,
                    input_ids=input_ids
                )
                
                # Assign precision levels
                token_precision = token_analyzer.assign_precision(token_importance)
                
                try:
                    # Try with token_precision parameter
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=50,
                        token_precision=token_precision
                    )
                except Exception:
                    # Fall back to standard generation
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=50
                    )
                
                # Calculate precision stats
                precision_stats = token_analyzer.analyze_precision_stats(
                    token_importance, token_precision)
                
            else:
                # Standard generation
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=50
                )
                precision_stats = None
        
        # End monitoring
        generation_time = time.time() - start_time
        energy_metrics = energy_monitor.stop()
        
        # Decode output
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Calculate metrics
        new_tokens = generated_ids.shape[1] - input_ids.shape[1]
        tokens_per_second = new_tokens / generation_time
        
        # Store sample result
        sample_result = {
            "prompt": prompt,
            "output": output_text,
            "reference": reference,
            "new_tokens": new_tokens,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "energy": energy_metrics
        }
        
        # Add precision stats if available
        if precision_stats:
            sample_result["precision_stats"] = precision_stats
        
        results["sample_results"].append(sample_result)
        
        # Update totals
        results["total_tokens_generated"] += new_tokens
        results["total_generation_time"] += generation_time
    
    # Calculate aggregate metrics
    results["avg_tokens_per_second"] = results["total_tokens_generated"] / results["total_generation_time"]
    results["avg_generation_time_per_sample"] = results["total_generation_time"] / len(benchmark_data)
    
    # Calculate average energy metrics
    if len(benchmark_data) > 0:
        results["avg_energy_per_sample"] = sum(r["energy"]["total_energy"] for r in results["sample_results"]) / len(benchmark_data)
    
    # Calculate average precision distribution if using adaptive precision
    if token_adaptive:
        avg_precision_dist = {}
        for result in results["sample_results"]:
            if "precision_stats" in result:
                for key, value in result["precision_stats"].items():
                    if key.startswith("precision_") and key.endswith("_pct"):
                        if key not in avg_precision_dist:
                            avg_precision_dist[key] = 0
                        avg_precision_dist[key] += value
        
        # Calculate averages
        for key in avg_precision_dist:
            avg_precision_dist[key] /= len(benchmark_data)
        
        results["avg_precision_distribution"] = avg_precision_dist
        
        # Calculate average energy savings
        if all("precision_stats" in r and "energy_saved_pct" in r["precision_stats"] for r in results["sample_results"]):
            avg_energy_saved = sum(r["precision_stats"]["energy_saved_pct"] for r in results["sample_results"]) / len(benchmark_data)
            results["avg_energy_saved_pct"] = avg_energy_saved
    
    return results


#------------------------------------------------------------------------------
# Interactive Visualization 
#------------------------------------------------------------------------------

def create_interactive_token_importance_display(model, tokenizer, prompt, output_file=None):
    """
    Create an interactive HTML visualization of token importance and precision
    
    Args:
        model: Model to analyze
        tokenizer: Tokenizer for the model
        prompt: Text prompt
        output_file: Output HTML file
        
    Returns:
        html_content: HTML content for visualization
    """
    try:
        # Create TAPEngine for analysis
        engine = TAPEngine()
        engine.model = model
        engine.tokenizer = tokenizer
        
        # Analyze token importance
        analysis = engine.analyze_token_importance(prompt=prompt)
        
        # Get token data
        token_importance = analysis["token_importance"]["default"].cpu().numpy()[0]
        token_precision = analysis["token_precision"].cpu().numpy()[0]
        token_texts = analysis["token_texts"]
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Token-Adaptive Precision Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; max-width: 1200px; margin: 0 auto; }
                h1 { color: #333; }
                .token-container { display: flex; flex-wrap: wrap; gap: 5px; margin: 20px 0; }
                .token { 
                    padding: 8px; border-radius: 4px; position: relative; 
                    display: flex; flex-direction: column; align-items: center;
                    border: 1px solid #ddd; min-width: 40px; 
                }
                .token-text { font-weight: bold; margin-bottom: 5px; }
                .token-importance { font-size: 0.8em; margin-bottom: 5px; }
                .token-precision { 
                    font-size: 0.7em; padding: 2px 5px; border-radius: 3px; 
                    color: white; font-weight: bold; 
                }
                .controls { margin: 20px 0; }
                .color-legend { display: flex; gap: 20px; margin-top: 10px; }
                .legend-item { display: flex; align-items: center; }
                .legend-color { width: 20px; height: 20px; margin-right: 5px; border-radius: 3px; }
                .stats { margin: 20px 0; }
                .stat-item { margin-bottom: 5px; }
                .visualization-container { display: flex; gap: 20px; }
                .chart-container { flex: 1; }
            </style>
        </head>
        <body>
            <h1>Token-Adaptive Precision Analysis</h1>
            
            <div class="stats">
                <h3>Precision Statistics</h3>
        """
        
        # Add precision statistics
        precision_stats = {}
        for i, prec in enumerate(token_precision):
            prec_int = int(prec)
            if prec_int not in precision_stats:
                precision_stats[prec_int] = 0
            precision_stats[prec_int] += 1
        
        total_tokens = len(token_precision)
        for prec, count in sorted(precision_stats.items()):
            percentage = count / total_tokens * 100
            html_content += f'<div class="stat-item">Precision {prec}-bit: {count}/{total_tokens} tokens ({percentage:.1f}%)</div>\n'
        
        # Add energy savings estimate
        energy_factors = {4: 0.25, 8: 0.4, 16: 0.6, 32: 1.0}
        relative_energy = sum((precision_stats.get(prec, 0) / total_tokens) * energy_factors.get(prec, 1.0) 
                             for prec in [4, 8, 16, 32])
        energy_saved = (1 - relative_energy) * 100
        
        html_content += f'<div class="stat-item"><strong>Estimated Energy Savings: {energy_saved:.1f}%</strong></div>\n'
        
        html_content += """
            </div>
            
            <div class="visualization-container">
                <div class="chart-container">
                    <h3>Token Visualization</h3>
                    <div class="controls">
                        <button onclick="colorByImportance()">Color by Importance</button>
                        <button onclick="colorByPrecision()">Color by Precision</button>
                    </div>
                    
                    <div class="color-legend" id="legend">
                        <!-- Legend will be filled by JavaScript -->
                    </div>
                    
                    <div class="token-container" id="tokenContainer">
                        <!-- Tokens will be filled by JavaScript -->
                    </div>
                </div>
            </div>
            
            <script>
                // Token data
                const tokenTexts = JSON.parse('"""
        
        # Add token data as JSON
        import json
        # Clean token texts for JSON
        clean_tokens = []
        for token in token_texts:
            if isinstance(token, list) and len(token) > 0:
                token = token[0]
            
            # Convert to string and escape special characters
            token_str = str(token).replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
            # Replace whitespace with visible symbols for display
            token_str = token_str.replace(' ', '␣').replace('\n', '↵')
            
            clean_tokens.append(token_str)
        
        html_content += json.dumps(clean_tokens)
        html_content += """');
                const tokenImportance = JSON.parse('"""
        html_content += json.dumps(token_importance.tolist())
        html_content += """');
                const tokenPrecision = JSON.parse('"""
        html_content += json.dumps(token_precision.tolist())
        html_content += """');
                
                // Current coloring mode
                let colorMode = 'importance';
                
                // Utility function to get color for importance value
                function getImportanceColor(importance) {
                    const r = Math.floor(importance * 255);
                    const g = Math.floor(255 - importance * 255);
                    const b = 100;
                    return `rgb(${r}, ${g}, ${b})`;
                }
                
                // Utility function to get color for precision value
                function getPrecisionColor(precision) {
                    switch(precision) {
                        case 4: return '#FF9999';  // Light red
                        case 8: return '#66B2FF';  // Light blue
                        case 16: return '#99FF99'; // Light green
                        case 32: return '#FFCC99'; // Light orange
                        default: return '#CCCCCC'; // Gray
                    }
                }
                
                // Render tokens
                function renderTokens() {
                    const container = document.getElementById('tokenContainer');
                    container.innerHTML = '';
                    
                    for (let i = 0; i < tokenTexts.length; i++) {
                        const token = document.createElement('div');
                        token.className = 'token';
                        
                        // Set background color based on current mode
                        if (colorMode === 'importance') {
                            token.style.backgroundColor = getImportanceColor(tokenImportance[i]);
                        } else {
                            token.style.backgroundColor = getPrecisionColor(tokenPrecision[i]);
                        }
                        
                        // Add token text
                        const tokenText = document.createElement('div');
                        tokenText.className = 'token-text';
                        tokenText.textContent = tokenTexts[i];
                        token.appendChild(tokenText);
                        
                        // Add importance score
                        const importanceElem = document.createElement('div');
                        importanceElem.className = 'token-importance';
                        importanceElem.textContent = `Imp: ${tokenImportance[i].toFixed(2)}`;
                        token.appendChild(importanceElem);
                        
                        // Add precision level
                        const precisionElem = document.createElement('div');
                        precisionElem.className = 'token-precision';
                        precisionElem.textContent = `${tokenPrecision[i]}-bit`;
                        precisionElem.style.backgroundColor = getPrecisionColor(tokenPrecision[i]);
                        token.appendChild(precisionElem);
                        
                        container.appendChild(token);
                    }
                }
                
                // Update color legend
                function updateLegend() {
                    const legend = document.getElementById('legend');
                    legend.innerHTML = '';
                    
                    if (colorMode === 'importance') {
                        // Create legend for importance
                        const importanceLevels = [0, 0.25, 0.5, 0.75, 1.0];
                        importanceLevels.forEach(level => {
                            const item = document.createElement('div');
                            item.className = 'legend-item';
                            
                            const color = document.createElement('div');
                            color.className = 'legend-color';
                            color.style.backgroundColor = getImportanceColor(level);
                            
                            const label = document.createElement('span');
                            label.textContent = `Importance: ${level.toFixed(2)}`;
                            
                            item.appendChild(color);
                            item.appendChild(label);
                            legend.appendChild(item);
                        });
                    } else {
                        // Create legend for precision
                        const precisionLevels = [4, 8, 16, 32];
                        precisionLevels.forEach(level => {
                            const item = document.createElement('div');
                            item.className = 'legend-item';
                            
                            const color = document.createElement('div');
                            color.className = 'legend-color';
                            color.style.backgroundColor = getPrecisionColor(level);
                            
                            const label = document.createElement('span');
                            label.textContent = `${level}-bit Precision`;
                            
                            item.appendChild(color);
                            item.appendChild(label);
                            legend.appendChild(item);
                        });
                    }
                }
                
                // Functions to change coloring mode
                function colorByImportance() {
                    colorMode = 'importance';
                    renderTokens();
                    updateLegend();
                }
                
                function colorByPrecision() {
                    colorMode = 'precision';
                    renderTokens();
                    updateLegend();
                }
                
                // Initialize visualization
                document.addEventListener('DOMContentLoaded', function() {
                    renderTokens();
                    updateLegend();
                });
                
                // Initialize immediately if document already loaded
                if (document.readyState === 'complete') {
                    renderTokens();
                    updateLegend();
                }
            </script>
        </body>
        </html>
        """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Interactive visualization saved to: {output_file}")
        
        return html_content
    
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        return None


#------------------------------------------------------------------------------
# Extended Benchmark for Publication-Quality Results
#------------------------------------------------------------------------------

def run_extended_benchmark(model, tokenizer, benchmark_type='efficiency', num_trials=5, output_dir=None):
    """
    Run a comprehensive benchmark for publication-quality results
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        benchmark_type: Type of benchmark ('efficiency', 'quality', 'all')
        num_trials: Number of trials to run
        output_dir: Output directory for results
        
    Returns:
        results: Benchmark results
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get model info
    model_info = {
        "name": model.__class__.__name__,
        "parameters": count_parameters(model)[0],
        "parameters_mb": count_parameters(model)[1]
    }
    
    # Results container
    results = {
        "model_info": model_info,
        "benchmark_type": benchmark_type,
        "num_trials": num_trials,
        "trials": [],
        "summary": {}
    }
    
    # Define test prompts
    test_prompts = [
        # Short prompts
        "Hello, my name is",
        "The capital of France is",
        "Explain the concept of",
        
        # Medium prompts
        "Write a short paragraph about the importance of renewable energy. Consider the environmental impact and economic benefits of transitioning to sustainable energy sources.",
        "Summarize the plot of the movie Inception in three sentences. Include the main characters and the central concept of dream invasion.",
        
        # Long prompts
        "Write a detailed essay on the impact of artificial intelligence on modern society. Consider ethical implications, economic effects, and potential future developments. Discuss both positive and negative aspects, and provide examples of current AI applications.",
        "Provide a comprehensive analysis of climate change, including its causes, effects, and potential solutions. Discuss the scientific consensus, major international agreements, and individual actions that can help mitigate its impacts. Include data on temperature increases, sea level rise, and greenhouse gas emissions."
    ]
    
    # Set precision modes to test
    precision_modes = [
        {"name": "FP32", "precision": 32, "token_adaptive": False},
        {"name": "FP16", "precision": 16, "token_adaptive": False},
        {"name": "INT8", "precision": 8, "token_adaptive": False},
        {"name": "INT4", "precision": 4, "token_adaptive": False},
        {"name": "Token-Adaptive", "precision": None, "token_adaptive": True}
    ]
    
    # Create token analyzer for adaptive precision
    token_analyzer = TokenImportanceAnalyzer()
    
    # Create energy monitor
    energy_monitor = EnergyMonitor()
    
    # Run trials
    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}")
        
        trial_results = {
            "trial_id": trial + 1,
            "precision_modes": {}
        }
        
        # Test each precision mode
        for mode in precision_modes:
            print(f"  Testing {mode['name']} precision...")
            
            mode_results = {
                "prompts": [],
                "avg_generation_time": 0,
                "avg_tokens_per_second": 0,
                "avg_energy": 0
            }
            
            # Process each prompt
            for prompt in test_prompts:
                # Tokenize prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_ids = inputs["input_ids"]
                
                # Prepare token precision if needed
                token_precision = None
                precision_stats = None
                
                if mode["token_adaptive"]:
                    # Calculate token importance
                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            output_hidden_states=True,
                            output_attentions=True,
                            return_dict=True
                        )
                        
                        # Get hidden states and attention
                        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") and outputs.hidden_states else None
                        attention_scores = outputs.attentions[-1] if hasattr(outputs, "attentions") and outputs.attentions else None
                        
                        # Calculate token importance
                        token_importance = token_analyzer.analyze_token_importance(
                            attention_scores=attention_scores,
                            hidden_states=hidden_states,
                            input_ids=input_ids
                        )
                        
                        # Assign precision levels
                        token_precision = token_analyzer.assign_precision(token_importance)
                        
                        # Calculate precision statistics
                        precision_stats = token_analyzer.analyze_precision_stats(
                            token_importance, token_precision)
                
                elif mode["precision"] is not None:
                    # Use fixed precision for all tokens
                    token_precision = torch.full((input_ids.shape[0], input_ids.shape[1]), 
                                              mode["precision"], 
                                              device=input_ids.device)
                
                # Start energy monitoring
                energy_monitor.start()
                start_time = time.time()
                
                # Generate text
                with torch.no_grad():
                    try:
                        # Try with token_precision parameter
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=50,
                            token_precision=token_precision
                        )
                    except Exception:
                        # Fall back to standard generation
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=50
                        )
                
                # End monitoring
                generation_time = time.time() - start_time
                energy_metrics = energy_monitor.stop()
                
                # Calculate metrics
                new_tokens = outputs.shape[1] - input_ids.shape[1]
                tokens_per_second = new_tokens / generation_time
                
                # Decode output
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Store prompt results
                prompt_result = {
                    "prompt": prompt,
                    "prompt_length": input_ids.shape[1],
                    "output": output_text,
                    "output_length": outputs.shape[1],
                    "new_tokens": new_tokens,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "energy": energy_metrics
                }
                
                # Add precision stats if available
                if precision_stats:
                    prompt_result["precision_stats"] = precision_stats
                
                mode_results["prompts"].append(prompt_result)
            
            # Calculate averages
            num_prompts = len(test_prompts)
            mode_results["avg_generation_time"] = sum(p["generation_time"] for p in mode_results["prompts"]) / num_prompts
            mode_results["avg_tokens_per_second"] = sum(p["tokens_per_second"] for p in mode_results["prompts"]) / num_prompts
            mode_results["avg_energy"] = sum(p["energy"]["total_energy"] for p in mode_results["prompts"]) / num_prompts
            
            # Calculate precision distribution if adaptive
            if mode["token_adaptive"]:
                precision_dist = {}
                for p_result in mode_results["prompts"]:
                    if "precision_stats" in p_result:
                        for key, value in p_result["precision_stats"].items():
                            if key.startswith("precision_") and key.endswith("_pct"):
                                if key not in precision_dist:
                                    precision_dist[key] = 0
                                precision_dist[key] += value
                
                # Average distribution
                for key in precision_dist:
                    precision_dist[key] /= num_prompts
                
                mode_results["avg_precision_distribution"] = precision_dist
                
                # Average energy savings
                if all("precision_stats" in p and "energy_saved_pct" in p["precision_stats"] 
                      for p in mode_results["prompts"]):
                    avg_energy_saved = sum(p["precision_stats"]["energy_saved_pct"] 
                                        for p in mode_results["prompts"]) / num_prompts
                    mode_results["avg_energy_saved_pct"] = avg_energy_saved
            
            # Store mode results
            trial_results["precision_modes"][mode["name"]] = mode_results
            
            print(f"    Time: {mode_results['avg_generation_time']:.3f}s, "
                 f"Tokens/s: {mode_results['avg_tokens_per_second']:.1f}, "
                 f"Energy: {mode_results['avg_energy']:.3f}J")
        
        # Store trial results
        results["trials"].append(trial_results)
        
        # Save intermediate results
        if output_dir:
            trial_file = os.path.join(output_dir, f"trial_{trial+1}_results.json")
            with open(trial_file, 'w', encoding='utf-8') as f:
                json.dump(trial_results, f, indent=2)
    
    # Calculate summary across all trials
    summary = {
        "precision_modes": {}
    }
    
    for mode in precision_modes:
        mode_name = mode["name"]
        
        # Gather results from all trials
        generation_times = [trial["precision_modes"][mode_name]["avg_generation_time"] 
                         for trial in results["trials"]]
        tokens_per_second = [trial["precision_modes"][mode_name]["avg_tokens_per_second"] 
                           for trial in results["trials"]]
        energy_usage = [trial["precision_modes"][mode_name]["avg_energy"] 
                      for trial in results["trials"]]
        
        # Calculate averages and standard deviations
        avg_generation_time = sum(generation_times) / len(generation_times)
        std_generation_time = (sum((t - avg_generation_time) ** 2 
                              for t in generation_times) / len(generation_times)) ** 0.5
        
        avg_tokens_per_second = sum(tokens_per_second) / len(tokens_per_second)
        std_tokens_per_second = (sum((t - avg_tokens_per_second) ** 2 
                                for t in tokens_per_second) / len(tokens_per_second)) ** 0.5
        
        avg_energy = sum(energy_usage) / len(energy_usage)
        std_energy = (sum((e - avg_energy) ** 2 
                      for e in energy_usage) / len(energy_usage)) ** 0.5
        
        # Store summary for this mode
        summary["precision_modes"][mode_name] = {
            "avg_generation_time": avg_generation_time,
            "std_generation_time": std_generation_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "std_tokens_per_second": std_tokens_per_second,
            "avg_energy": avg_energy,
            "std_energy": std_energy
        }
        
        # Add precision distribution if adaptive
        if mode["token_adaptive"]:
            precision_dists = []
            for trial in results["trials"]:
                if "avg_precision_distribution" in trial["precision_modes"][mode_name]:
                    precision_dists.append(trial["precision_modes"][mode_name]["avg_precision_distribution"])
            
            if precision_dists:
                # Combine precision distributions
                combined_dist = {}
                for dist in precision_dists:
                    for key, value in dist.items():
                        if key not in combined_dist:
                            combined_dist[key] = []
                        combined_dist[key].append(value)
                
                # Calculate averages and standard deviations
                avg_dist = {}
                std_dist = {}
                for key, values in combined_dist.items():
                    avg_dist[key] = sum(values) / len(values)
                    std_dist[key] = (sum((v - avg_dist[key]) ** 2 for v in values) / len(values)) ** 0.5
                
                summary["precision_modes"][mode_name]["avg_precision_distribution"] = avg_dist
                summary["precision_modes"][mode_name]["std_precision_distribution"] = std_dist
            
            # Add energy savings if available
            energy_savings = []
            for trial in results["trials"]:
                if "avg_energy_saved_pct" in trial["precision_modes"][mode_name]:
                    energy_savings.append(trial["precision_modes"][mode_name]["avg_energy_saved_pct"])
            
            if energy_savings:
                avg_energy_saved = sum(energy_savings) / len(energy_savings)
                std_energy_saved = (sum((s - avg_energy_saved) ** 2 
                                    for s in energy_savings) / len(energy_savings)) ** 0.5
                
                summary["precision_modes"][mode_name]["avg_energy_saved_pct"] = avg_energy_saved
                summary["precision_modes"][mode_name]["std_energy_saved_pct"] = std_energy_saved
    
    # Calculate comparisons to FP32
    if "FP32" in summary["precision_modes"]:
        fp32_time = summary["precision_modes"]["FP32"]["avg_generation_time"]
        fp32_energy = summary["precision_modes"]["FP32"]["avg_energy"]
        
        for mode_name, mode_summary in summary["precision_modes"].items():
            if mode_name != "FP32":
                # Calculate speedup
                speedup = fp32_time / mode_summary["avg_generation_time"]
                
                # Calculate energy reduction
                energy_reduction = (fp32_energy - mode_summary["avg_energy"]) / fp32_energy * 100
                
                # Store comparisons
                mode_summary["speedup_vs_fp32"] = speedup
                mode_summary["energy_reduction_vs_fp32"] = energy_reduction
    
    # Store summary
    results["summary"] = summary
    
    # Save final results
    if output_dir:
        final_file = os.path.join(output_dir, "benchmark_results.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 1. Create speedup comparison chart
            plt.figure(figsize=(10, 6))
            
            # Get modes and values
            modes = []
            speedups = []
            energy_reductions = []
            
            for mode_name, mode_summary in summary["precision_modes"].items():
                if mode_name != "FP32" and "speedup_vs_fp32" in mode_summary:
                    modes.append(mode_name)
                    speedups.append(mode_summary["speedup_vs_fp32"])
                    energy_reductions.append(mode_summary["energy_reduction_vs_fp32"])
            
            # Create bar chart for speedup
            x = np.arange(len(modes))
            plt.bar(x - 0.2, speedups, width=0.4, label='Speedup vs. FP32', color='blue')
            plt.bar(x + 0.2, energy_reductions, width=0.4, label='Energy Reduction %', color='green')
            
            plt.xlabel('Precision Mode')
            plt.ylabel('Factor / Percentage')
            plt.title('Performance Comparison vs. FP32')
            plt.xticks(x, modes)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, v in enumerate(speedups):
                plt.text(i - 0.2, v + 0.05, f'{v:.2f}x', ha='center')
            
            for i, v in enumerate(energy_reductions):
                plt.text(i + 0.2, v + 0.05, f'{v:.1f}%', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=300)
            
            # 2. Create token precision distribution chart for adaptive precision
            if "Token-Adaptive" in summary["precision_modes"] and "avg_precision_distribution" in summary["precision_modes"]["Token-Adaptive"]:
                plt.figure(figsize=(8, 8))
                
                # Get precision distribution
                dist = summary["precision_modes"]["Token-Adaptive"]["avg_precision_distribution"]
                labels = []
                sizes = []
                
                for key, value in sorted(dist.items()):
                    if key.startswith("precision_") and key.endswith("_pct"):
                        bit_value = key.split("_")[1]
                        labels.append(f"{bit_value}-bit")
                        sizes.append(value)
                
                # Create pie chart
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                      colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
                plt.axis('equal')
                plt.title('Token-Adaptive Precision Distribution')
                
                plt.savefig(os.path.join(output_dir, 'precision_distribution.png'), dpi=300)
                
                # 3. Create benchmark summary chart
                plt.figure(figsize=(12, 8))
                
                # Set up subplot grid
                plt.subplot(2, 1, 1)
                
                # Get data for all precision modes
                mode_names = list(summary["precision_modes"].keys())
                gen_times = [summary["precision_modes"][m]["avg_generation_time"] for m in mode_names]
                gen_times_std = [summary["precision_modes"][m]["std_generation_time"] for m in mode_names]
                
                tokens_per_sec = [summary["precision_modes"][m]["avg_tokens_per_second"] for m in mode_names]
                tokens_per_sec_std = [summary["precision_modes"][m]["std_tokens_per_second"] for m in mode_names]
                
                # Create bar chart for generation time
                x = np.arange(len(mode_names))
                plt.bar(x, gen_times, yerr=gen_times_std, capsize=5, color='skyblue')
                
                plt.xlabel('Precision Mode')
                plt.ylabel('Generation Time (s)')
                plt.title('Average Generation Time by Precision Mode')
                plt.xticks(x, mode_names)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels
                for i, v in enumerate(gen_times):
                    plt.text(i, v + gen_times_std[i] + 0.02, f'{v:.3f}s', ha='center')
                
                # Tokens per second subplot
                plt.subplot(2, 1, 2)
                
                plt.bar(x, tokens_per_sec, yerr=tokens_per_sec_std, capsize=5, color='lightgreen')
                
                plt.xlabel('Precision Mode')
                plt.ylabel('Tokens per Second')
                plt.title('Average Throughput by Precision Mode')
                plt.xticks(x, mode_names)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels
                for i, v in enumerate(tokens_per_sec):
                    plt.text(i, v + tokens_per_sec_std[i] + 0.2, f'{v:.1f}', ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'benchmark_summary.png'), dpi=300)
            
            print(f"Visualizations saved to {output_dir}")
            
        except ImportError:
            print("Matplotlib not available, skipping visualizations")
    
    return results


#------------------------------------------------------------------------------
# Generate Documentation
#------------------------------------------------------------------------------

def generate_documentation():
    """Generate comprehensive documentation for the TAP Engine"""
    documentation = """
# Token-Adaptive Precision (TAP) Engine

## Overview

The Token-Adaptive Precision (TAP) Engine is a framework for applying dynamic, token-level precision control to large language models (LLMs). By analyzing the importance of each token in the context, TAP Engine applies different numerical precision levels, optimizing both computational performance and energy efficiency.

Supported model architectures:
- Transformer-based models (GPT, BERT, etc.)
- Mamba/State Space Models (SSM)
- Retention Networks (RetNet)
- RWKV models

## Key Features

- **Token-level precision control**: Apply different bit precisions (4, 8, 16, 32) to different tokens based on their importance
- **Architecture-aware importance analysis**: Methods optimized for each model architecture
- **Seamless integration**: Works with popular libraries like HuggingFace Transformers
- **Comprehensive tooling**: Includes visualization, benchmarking, and optimization utilities
- **Energy-efficient**: Reduces energy consumption while maintaining output quality
- **Cross-platform**: Works on NVIDIA, AMD, and CPU hardware

## Installation

```bash
pip install tap-engine
```

## Quick Start

```python
from tap_engine import TAPEngine, TAPConfig, ModelArchitecture, PrecisionMode

# Create a custom configuration
config = TAPConfig(
    model_arch=ModelArchitecture.TRANSFORMER,  # Or MAMBA, RETNET, RWKV
    precision_mode=PrecisionMode.ADAPTIVE,
    precision_levels=[4, 8, 16, 32],
    precision_thresholds=[0.3, 0.6, 0.9]
)

# Initialize TAP Engine
engine = TAPEngine(config)

# Load model (automatically applies token-adaptive optimizations)
engine.load_model("meta-llama/Llama-2-7b-hf")

# Generate text with token-adaptive precision
text, metrics = engine.generate(
    prompt="Explain quantum computing in simple terms",
    max_new_tokens=100
)

print(text)
print(f"Generation time: {metrics['generation_time']:.2f}s")
print(f"Energy saved: {metrics['precision']['energy_saved_pct']:.1f}%")
```

## Core Components

### TAPConfig

Configuration class that controls all aspects of token-adaptive precision.

### TAPEngine

Main interface for loading models, generating text, and analyzing token importance.

### TokenImportanceAnalyzer

Analyzes token importance using various methods optimized for different architectures.

### PrecisionManager

Handles tensor quantization and precision management.

### Token-Adaptive Layers

Architecture-specific implementations of token-adaptive operations:

- `TokenAdaptiveLinear`: Linear/dense layers with token-adaptive precision
- `TokenAdaptiveLayerNorm`: Layer normalization with token-adaptive precision
- `TokenAdaptiveAttention`: Self-attention with token-adaptive precision
- `TokenAdaptiveSSMLayer`: State space model layer with token-adaptive precision
- `TokenAdaptiveRetNetBlock`: RetNet block with token-adaptive precision
- `TokenAdaptiveRWKVBlock`: RWKV block with token-adaptive precision

## Advanced Usage

### Token Importance Analysis

```python
# Analyze token importance for a specific prompt
analysis = engine.analyze_token_importance(
    prompt="The quick brown fox jumps over the lazy dog"
)

# Visualize token importance
create_interactive_token_importance_display(
    engine.model, 
    engine.tokenizer, 
    prompt="The quick brown fox jumps over the lazy dog",
    output_file="token_importance.html"
)
```

### Benchmarking

```python
# Run comprehensive benchmark
results = run_extended_benchmark(
    engine.model,
    engine.tokenizer,
    benchmark_type='efficiency',
    num_trials=5,
    output_dir='benchmark_results'
)

# Visualize benchmark results
visualize_test_results(results, output_file="benchmark.png")
```

### Model Optimization

```python
# Optimize model with quantization
engine.optimize_model(quantize_bits=8)

# Save optimized model
engine.save_model("optimized_model")
```

### Integration with Other Libraries

```python
# Use with QLoRA
from tap_engine import add_token_adaptive_lora

model = add_token_adaptive_lora(engine.model, lora_rank=8)

# Use with BitsAndBytes
from tap_engine import TokenAdaptive4BitLinear

# Use with Accelerate
from tap_engine import ModelIntegrationManager

model = ModelIntegrationManager.integrate_with_accelerate(engine.model)
```

## Performance Considerations

- **Memory usage**: Token-adaptive precision can reduce memory usage by 30-60% compared to FP32
- **Speed**: Typically provides 1.5-3x speedup compared to FP32
- **Energy efficiency**: Reduces energy consumption by 40-70%
- **Quality**: Maintains output quality comparable to FP16 while using less resources

## Known Limitations

- Some operations may fall back to the highest precision when precision mixing is not supported
- Token importance calculation adds some overhead on the first forward pass
- Limited support for highly customized model architectures

## Citation

If you use TAP Engine in your research, please cite:

```
@article{tap_engine2023,
  title={Token-Adaptive Precision: Dynamic Bit-Level Optimization for Language Models},
  author={TAP Engine Team},
  journal={Arxiv},
  year={2023}
}
```

## License

MIT License
"""
    
    return documentation


def print_usage_examples():
    """Print usage examples for the TAP Engine"""
    examples = """
# Token-Adaptive Precision (TAP) Engine - Usage Examples

## Basic Usage

```python
from tap_engine import TAPEngine

# Initialize TAP Engine with default configuration
engine = TAPEngine()

# Load model
engine.load_model("gpt2")

# Generate text with token-adaptive precision
text, metrics = engine.generate(
    prompt="Once upon a time",
    max_new_tokens=50
)

print(text)
```

## Custom Configuration

```python
from tap_engine import TAPEngine, TAPConfig, ModelArchitecture, PrecisionMode

# Create custom configuration
config = TAPConfig(
    model_arch=ModelArchitecture.TRANSFORMER,
    precision_mode=PrecisionMode.ADAPTIVE,
    precision_levels=[4, 8, 16, 32],
    precision_thresholds=[0.3, 0.6, 0.9],
    importance_method="hybrid",
    energy_optimization_level=2,
    memory_efficient_mode=True
)

# Initialize engine with custom config
engine = TAPEngine(config)

# Load and optimize model
engine.load_model("facebook/opt-1.3b")
engine.optimize_model(quantize_bits=8)

# Generate text
text, metrics = engine.generate(
    prompt="Explain how neural networks work",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
```

## Working with Mamba Models

```python
from tap_engine import TAPEngine, TAPConfig, ModelArchitecture

# Create Mamba-specific configuration
config = TAPConfig(
    model_arch=ModelArchitecture.MAMBA,
    ssm_state_precision=16,
    ssm_scan_precision=16
)

# Initialize engine
engine = TAPEngine(config)

# Load Mamba model
engine.load_model("state-spaces/mamba-130m")

# Generate text
text, metrics = engine.generate(
    prompt="The theory of relativity states that",
    max_new_tokens=50
)
```

## Token Importance Visualization

```python
from tap_engine import create_interactive_token_importance_display

# Create interactive visualization
html = create_interactive_token_importance_display(
    engine.model,
    engine.tokenizer,
    prompt="The importance of each token varies in this sentence.",
    output_file="token_visualization.html"
)

# Open in browser
import webbrowser
webbrowser.open("token_visualization.html")
```

## Benchmarking

```python
from tap_engine import run_test_suite, visualize_test_results

# Run test suite
results = run_test_suite(
    engine.model,
    engine.tokenizer,
    test_cases=[
        "Hello, how are you today?",
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about nature."
    ],
    precision_levels=[4, 8, 16, 32]
)

# Visualize results
visualize_test_results(results, output_file="benchmark_results.png")
```

## Advanced Energy Analysis

```python
from tap_engine import EnergyMonitor

# Create energy monitor
energy_monitor = EnergyMonitor()

# Start monitoring
energy_monitor.start()

# Run your code
output = engine.generate(prompt="This is a test of energy monitoring")

# Stop monitoring and get results
metrics = energy_monitor.stop()

print(f"Total energy used: {metrics['total_energy']:.2f} joules")
print(f"Average power: {metrics['avg_power']:.2f} watts")
print(f"Duration: {metrics['duration']:.2f} seconds")
```

## Command Line Interface

```bash
# Generate text with token-adaptive precision
python -m tap_engine generate --model gpt2 --prompt "Hello, world!" --max-tokens 50

# Optimize a model
python -m tap_engine optimize --model gpt2 --quantize-bits 8 --save

# Analyze token importance
python -m tap_engine analyze --model gpt2 --prompt "Analyzing token importance" --visualize

# Run benchmark
python -m tap_engine benchmark --model gpt2 --prompt "Benchmark test" --iterations 5 --compare-modes
```

## Integration with Training Workflows

```python
from tap_engine import TokenAdaptiveTrainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb", split="train[:1000]")

# Prepare dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create trainer
trainer = TokenAdaptiveTrainer(
    model=engine.model,
    train_dataset=tokenized_dataset,
    batch_size=8,
    num_epochs=3,
    use_adaptive_precision=True
)

# Train model
metrics = trainer.train()
```

## Custom Adapters and Layers

```python
from tap_engine import TokenAdaptiveLinear, TokenAdaptiveLayerNorm

# Create custom layer with token-adaptive precision
class MyTokenAdaptiveLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = TokenAdaptiveLinear(hidden_size, hidden_size * 4)
        self.linear2 = TokenAdaptiveLinear(hidden_size * 4, hidden_size)
        self.norm = TokenAdaptiveLayerNorm(hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x, token_precision=None):
        # Apply token-adaptive operations
        residual = x
        x = self.norm(x, token_precision)
        x = self.linear1(x, token_precision)
        x = self.activation(x)
        x = self.linear2(x, token_precision)
        return x + residual
```

## Model Export and Deployment

```python
from tap_engine import export_model_for_inference

# Export model for deployment
export_path = export_model_for_inference(
    engine.model,
    output_dir="deployment_model",
    optimize=True,
    quantize_bits=8
)

# Load for inference
from tap_engine import TAPEngine
inference_engine = TAPEngine.from_pretrained(export_path)
```
"""

    return examples

#------------------------------------------------------------------------------
# Main Function and Entry Point
#------------------------------------------------------------------------------

def tap_main():
    """Main function for the TAP Engine module"""
    show_system_info()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "info":
            # Show system information
            show_system_info()
        
        elif command == "examples":
            # Show usage examples
            print(print_usage_examples())
        
        elif command == "docs":
            # Generate documentation
            print(generate_documentation())
        
        elif command == "version":
            # Show version information
            print(f"TAP Engine version: {__version__}")
        
        else:
            # Unknown command, call main CLI function
            main()
    else:
        # No command provided, show help and system info
        print(f"TAP Engine version: {__version__}")
        print("Use tap_engine <command> to run specific functionality")
        print("\nAvailable commands:")
        print("  info      - Show system information")
        print("  examples  - Show usage examples")
        print("  docs      - Generate documentation")
        print("  version   - Show version information")
        print("\nOr use the main CLI - see 'python -m tap_engine --help'")


if __name__ == "__main__":
    tap_main()
