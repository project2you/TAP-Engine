import psutil
import time
import threading
import datetime
import pandas as pd
import numpy as np
import os
import platform
import json
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

@dataclass
class HardwareProfile:
    """Hardware profile with power consumption characteristics."""
    cpu_tdp: float  # Thermal Design Power in watts
    gpu_tdp: float  # GPU TDP in watts
    ram_per_gb: float  # Power per GB RAM in watts
    disk_active: float  # Active power in watts
    disk_idle: float  # Idle power in watts
    baseline: float  # Baseline system power in watts
    
    @classmethod
    def detect_system(cls) -> 'HardwareProfile':
        """Auto-detect system and return appropriate hardware profile."""
        system = platform.system()
        machine = platform.machine()
        
        # Default values for a standard laptop
        profile = {
            'cpu_tdp': 45.0,
            'gpu_tdp': 75.0,
            'ram_per_gb': 0.375,
            'disk_active': 5.5,
            'disk_idle': 0.8,
            'baseline': 10.0
        }
        
        try:
            # Try to get CPU info
            if system == "Linux":
                # Try to read CPU TDP from Linux
                try:
                    with open('/sys/class/thermal/thermal_zone0/device/power/power_cap_uw', 'r') as f:
                        tdp_uw = int(f.read().strip())
                        profile['cpu_tdp'] = tdp_uw / 1000000  # Convert to watts
                except:
                    pass
            elif system == "Darwin":  # macOS
                # Use higher efficiency values for Apple Silicon
                if machine == "arm64":
                    profile['cpu_tdp'] = 20.0
                    profile['baseline'] = 5.0
            
            # Try to detect GPU
            try:
                if system == "Windows":
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        profile['gpu_tdp'] = max(g.powerDraw if hasattr(g, 'powerDraw') else 75.0 for g in gpus)
            except:
                pass
                
        except Exception as e:
            print(f"Could not auto-detect hardware profile: {e}")
            print("Using default values")
        
        return cls(**profile)

class RegionalCarbonIntensity:
    """Regional carbon intensity data."""
    
    # Default carbon intensity by region in gCO2/kWh
    DEFAULT_INTENSITIES = {
        'global': 475.0,
        'us': 379.0,
        'eu': 231.0,
        'china': 548.0,
        'india': 708.0,
        'uk': 180.0,
        'france': 56.0,
        'germany': 335.0,
        'canada': 120.0,
        'australia': 700.0,
        'nordic': 45.0,  # Nordic countries average
    }
    
    def __init__(self, region: str = 'global'):
        self.region = region.lower()
        
    def get_intensity(self) -> float:
        """Get carbon intensity for the region in gCO2/kWh."""
        return self.DEFAULT_INTENSITIES.get(self.region, self.DEFAULT_INTENSITIES['global'])
    
    @staticmethod
    def available_regions() -> List[str]:
        """Return list of available regions."""
        return list(RegionalCarbonIntensity.DEFAULT_INTENSITIES.keys())

class AIWorkloadProfiler:
    """Profiles for different AI workload types."""
    
    # Workload profiles with relative resource multipliers
    PROFILES = {
        'llm_inference': {
            'description': 'Large Language Model Inference',
            'cpu_multiplier': 0.8,
            'gpu_multiplier': 1.0,
            'ram_multiplier': 1.2,
            'io_multiplier': 0.3,
        },
        'llm_training': {
            'description': 'Large Language Model Training',
            'cpu_multiplier': 0.9,
            'gpu_multiplier': 1.6,
            'ram_multiplier': 1.4,
            'io_multiplier': 0.6,
        },
        'vision_inference': {
            'description': 'Computer Vision Inference',
            'cpu_multiplier': 0.6,
            'gpu_multiplier': 0.9,
            'ram_multiplier': 0.7,
            'io_multiplier': 0.4,
        },
        'vision_training': {
            'description': 'Computer Vision Model Training',
            'cpu_multiplier': 0.7,
            'gpu_multiplier': 1.4,
            'ram_multiplier': 0.8,
            'io_multiplier': 0.5,
        },
        'nlp_inference': {
            'description': 'Natural Language Processing Inference',
            'cpu_multiplier': 0.75,
            'gpu_multiplier': 0.8,
            'ram_multiplier': 0.9,
            'io_multiplier': 0.3,
        },
        'data_preprocessing': {
            'description': 'Data Preprocessing Pipeline',
            'cpu_multiplier': 1.0,
            'gpu_multiplier': 0.2,
            'ram_multiplier': 1.0,
            'io_multiplier': 1.2,
        },
    }
    
    @staticmethod
    def get_profile(workload_type: str) -> Dict[str, float]:
        """Get the profile for a specific workload type."""
        return AIWorkloadProfiler.PROFILES.get(workload_type, AIWorkloadProfiler.PROFILES['llm_inference'])
    
    @staticmethod
    def available_profiles() -> List[str]:
        """Return list of available profiles."""
        return list(AIWorkloadProfiler.PROFILES.keys())
    
    @staticmethod
    def profile_description(workload_type: str) -> str:
        """Get description of a workload profile."""
        profile = AIWorkloadProfiler.PROFILES.get(workload_type)
        return profile['description'] if profile else "Unknown workload type"

class ResourceTracker:
    def __init__(
        self, 
        interval_sec: int = 1,
        hardware_profile: Optional[HardwareProfile] = None,
        region: str = 'global',
        gpu_enabled: bool = False,
        detailed_gpu: bool = False,
        track_network: bool = True,
        track_processes: bool = True,
        output_dir: Optional[str] = None
    ):
        self.interval = interval_sec
        self.running = False
        self.data = []
        self.sections = []
        self.current_section = None
        self.current_workload_type = None
        self.section_start_time = None
        self.gpu_enabled = gpu_enabled
        self.detailed_gpu = detailed_gpu and gpu_enabled
        self.track_network = track_network
        self.track_processes = track_processes
        self.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Set hardware profile
        self.hardware_profile = hardware_profile or HardwareProfile.detect_system()
        
        # Set region for carbon intensity
        self.carbon_intensity = RegionalCarbonIntensity(region)
        
        # PUE (Power Usage Effectiveness) - datacenter overhead
        self.pue = 1.5  # Default value for modern data centers
        
        # Initialize GPU monitoring if enabled
        self.gpu_info = None
        if self.gpu_enabled:
            try:
                self.setup_gpu_monitoring()
            except Exception as e:
                print(f"Could not initialize GPU monitoring: {e}")
                self.gpu_enabled = False
        
        # Setup output directory
        self.output_dir = output_dir
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def setup_gpu_monitoring(self):
        """Setup GPU monitoring based on available libraries."""
        try:
            import GPUtil
            self.gpu_lib = "GPUtil"
            self.gpus = GPUtil.getGPUs()
            if not self.gpus:
                self.gpu_enabled = False
        except ImportError:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_lib = "pynvml"
                self.device_count = pynvml.nvmlDeviceGetCount()
                if self.device_count == 0:
                    self.gpu_enabled = False
            except ImportError:
                self.gpu_enabled = False
    
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization and memory metrics."""
        if not self.gpu_enabled:
            return {}
        
        try:
            if self.gpu_lib == "GPUtil":
                import GPUtil
                self.gpus = GPUtil.getGPUs()
                
                gpu_util = np.mean([gpu.load * 100 for gpu in self.gpus]) if self.gpus else 0
                gpu_mem_util = np.mean([gpu.memoryUtil * 100 for gpu in self.gpus]) if self.gpus else 0
                gpu_temp = np.mean([gpu.temperature for gpu in self.gpus]) if self.gpus else 0
                
                metrics = {
                    'gpu_percent': gpu_util,
                    'gpu_mem_percent': gpu_mem_util,
                    'gpu_temp': gpu_temp
                }
                
                # Add detailed per-GPU metrics if enabled
                if self.detailed_gpu and self.gpus:
                    for i, gpu in enumerate(self.gpus):
                        metrics[f'gpu{i}_percent'] = gpu.load * 100
                        metrics[f'gpu{i}_mem_percent'] = gpu.memoryUtil * 100
                        metrics[f'gpu{i}_temp'] = gpu.temperature
                
                return metrics
                
            elif self.gpu_lib == "pynvml":
                import pynvml
                
                gpu_metrics = []
                gpu_mem_metrics = []
                gpu_temp_metrics = []
                
                for i in range(self.device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_metrics.append(util.gpu)
                    
                    # Get memory utilization
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_util = (mem_info.used / mem_info.total) * 100
                    gpu_mem_metrics.append(mem_util)
                    
                    # Get temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temp_metrics.append(temp)
                
                metrics = {
                    'gpu_percent': np.mean(gpu_metrics) if gpu_metrics else 0,
                    'gpu_mem_percent': np.mean(gpu_mem_metrics) if gpu_mem_metrics else 0,
                    'gpu_temp': np.mean(gpu_temp_metrics) if gpu_temp_metrics else 0
                }
                
                # Add detailed per-GPU metrics if enabled
                if self.detailed_gpu and self.device_count > 0:
                    for i in range(self.device_count):
                        metrics[f'gpu{i}_percent'] = gpu_metrics[i]
                        metrics[f'gpu{i}_mem_percent'] = gpu_mem_metrics[i]
                        metrics[f'gpu{i}_temp'] = gpu_temp_metrics[i]
                
                return metrics
        
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
        
        return {}
    
    def get_network_metrics(self) -> Dict[str, float]:
        """Get network I/O metrics."""
        if not self.track_network:
            return {}
        
        try:
            net_io = psutil.net_io_counters()
            
            metrics = {
                'net_sent_mbps': 0,  # Will be calculated as delta
                'net_recv_mbps': 0,  # Will be calculated as delta
                'net_sent_bytes': net_io.bytes_sent,
                'net_recv_bytes': net_io.bytes_recv
            }
            
            # Calculate bandwidth if we have previous data
            if len(self.data) > 0:
                prev = self.data[-1]
                
                if 'net_sent_bytes' in prev and 'net_recv_bytes' in prev:
                    # Delta bytes
                    delta_sent = metrics['net_sent_bytes'] - prev['net_sent_bytes']
                    delta_recv = metrics['net_recv_bytes'] - prev['net_recv_bytes']
                    
                    # Convert to Mbps (megabits per second)
                    metrics['net_sent_mbps'] = (delta_sent * 8) / (1024 * 1024) / self.interval
                    metrics['net_recv_mbps'] = (delta_recv * 8) / (1024 * 1024) / self.interval
            
            return metrics
            
        except Exception as e:
            print(f"Error getting network metrics: {e}")
        
        return {}
    
    def get_process_metrics(self) -> Dict[str, float]:
        """Get process metrics related to AI workloads."""
        if not self.track_processes:
            return {}
        
        try:
            metrics = {
                'process_count': 0,
                'python_process_count': 0,
                'total_python_cpu_percent': 0,
                'total_python_memory_mb': 0
            }
            
            python_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                metrics['process_count'] += 1
                
                try:
                    # Check if it's a Python process potentially running AI code
                    is_python = False
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        is_python = True
                    elif proc.info['cmdline'] and any('python' in cmd.lower() for cmd in proc.info['cmdline'] if cmd):
                        is_python = True
                    
                    # If it's a Python process, get more details
                    if is_python:
                        metrics['python_process_count'] += 1
                        
                        # Sum CPU and memory usage
                        proc_cpu = proc.info['cpu_percent']
                        if proc_cpu is not None:
                            metrics['total_python_cpu_percent'] += proc_cpu
                        
                        if proc.info['memory_info'] is not None:
                            metrics['total_python_memory_mb'] += proc.info['memory_info'].rss / (1024 * 1024)
                        
                        # Collect for detailed reporting
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc_cpu,
                            'memory_mb': proc.info['memory_info'].rss / (1024 * 1024) if proc.info['memory_info'] else 0
                        })
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # Store top Python processes by CPU usage
            if python_processes:
                top_processes = sorted(python_processes, key=lambda x: x['cpu_percent'] if x['cpu_percent'] is not None else 0, reverse=True)[:5]
                for i, proc in enumerate(top_processes):
                    metrics[f'top_py_proc{i+1}_pid'] = proc['pid']
                    metrics[f'top_py_proc{i+1}_cpu'] = proc['cpu_percent']
                    metrics[f'top_py_proc{i+1}_memory_mb'] = proc['memory_mb']
            
            return metrics
            
        except Exception as e:
            print(f"Error getting process metrics: {e}")
        
        return {}
    
    def _collect_metrics(self):
        prev_disk_io = psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
        
        while self.running:
            timestamp = datetime.datetime.now()
            
            # Basic system metrics
            cpu = psutil.cpu_percent(interval=None)
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            
            # Memory metrics
            mem = psutil.virtual_memory()
            ram_percent = mem.percent
            ram_used_gb = mem.used / (1024 ** 3)
            
            # Disk metrics
            disk_percent = psutil.disk_usage('/').percent
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
            disk_read_mbps = 0
            disk_write_mbps = 0
            
            if disk_io and prev_disk_io:
                # Calculate disk I/O rates
                read_bytes_delta = disk_io.read_bytes - prev_disk_io.read_bytes
                write_bytes_delta = disk_io.write_bytes - prev_disk_io.write_bytes
                
                # Convert to MB/s
                disk_read_mbps = read_bytes_delta / (1024 * 1024) / self.interval
                disk_write_mbps = write_bytes_delta / (1024 * 1024) / self.interval
            
            prev_disk_io = disk_io
            
            # Get GPU metrics if enabled
            gpu_metrics = self.get_gpu_metrics() if self.gpu_enabled else {}
            
            # Get network metrics if enabled
            network_metrics = self.get_network_metrics() if self.track_network else {}
            
            # Get process metrics if enabled
            process_metrics = self.get_process_metrics() if self.track_processes else {}
            
            # Calculate power consumption
            power_metrics = self.estimate_power(
                cpu=cpu, 
                ram_gb=ram_used_gb,
                disk=disk_percent,
                disk_io_read_mbps=disk_read_mbps,
                disk_io_write_mbps=disk_write_mbps,
                **gpu_metrics
            )
            
            # Calculate carbon footprint
            carbon_metrics = self.estimate_carbon(power_metrics['power_watts'])
            
            # Apply workload-specific adjustments if a workload type is specified
            if self.current_workload_type:
                workload_metrics = self.apply_workload_profile(
                    self.current_workload_type,
                    power_metrics['power_watts']
                )
            else:
                workload_metrics = {
                    'workload_type': None,
                    'adjusted_power_watts': power_metrics['power_watts']
                }
            
            # Combine all metrics
            record = {
                'timestamp': timestamp,
                'cpu_percent': cpu,
                'ram_percent': ram_percent,
                'ram_used_gb': ram_used_gb,
                'disk_percent': disk_percent,
                'disk_read_mbps': disk_read_mbps,
                'disk_write_mbps': disk_write_mbps,
                'section': self.current_section,
                **gpu_metrics,
                **network_metrics,
                **process_metrics,
                **power_metrics,
                **carbon_metrics,
                **workload_metrics
            }
            
            # Add per-CPU metrics if available
            for i, cpu_val in enumerate(per_cpu):
                record[f'cpu{i}_percent'] = cpu_val
            
            self.data.append(record)
            
            # Sleep until next collection interval
            time.sleep(self.interval)
    
    def estimate_power(
        self, 
        cpu: float, 
        ram_gb: float,
        disk: float,
        disk_io_read_mbps: float = 0,
        disk_io_write_mbps: float = 0,
        gpu_percent: float = 0,
        **kwargs
    ) -> Dict[str, float]:
        """
        Estimate power consumption based on resource utilization.
        
        Args:
            cpu: CPU utilization percentage (0-100)
            ram_gb: RAM usage in GB
            disk: Disk utilization percentage (0-100)
            disk_io_read_mbps: Disk read rate in MB/s
            disk_io_write_mbps: Disk write rate in MB/s
            gpu_percent: GPU utilization percentage (0-100)
            **kwargs: Additional metrics not used in calculation
            
        Returns:
            Dictionary with power metrics
        """
        # CPU power calculation - non-linear scaling
        # At idle, CPUs use ~10-20% of TDP, at full load they use 100% of TDP
        idle_cpu_factor = 0.15  # 15% of TDP at idle
        cpu_power = self.hardware_profile.cpu_tdp * (idle_cpu_factor + (1 - idle_cpu_factor) * (cpu / 100))
        
        # RAM power
        ram_power = ram_gb * self.hardware_profile.ram_per_gb
        
        # Disk power - combination of utilization and I/O
        disk_idle_power = self.hardware_profile.disk_idle
        disk_active_power = self.hardware_profile.disk_active * (disk / 100)
        
        # Additional power for disk I/O
        disk_io_power = 0
        if disk_io_read_mbps > 0 or disk_io_write_mbps > 0:
            # Each 100 MB/s of I/O adds ~1W on SSDs
            disk_io_power = (disk_io_read_mbps + disk_io_write_mbps) * 0.01
        
        disk_power = disk_idle_power + disk_active_power + disk_io_power
        
        # GPU power if available
        gpu_power = 0
        if self.gpu_enabled and gpu_percent > 0:
            # Similar to CPU, GPUs use ~5-15% of TDP at idle
            idle_gpu_factor = 0.1  # 10% of TDP at idle
            gpu_power = self.hardware_profile.gpu_tdp * (idle_gpu_factor + (1 - idle_gpu_factor) * (gpu_percent / 100))
        
        # Other system components (motherboard, fans, etc.)
        baseline_power = self.hardware_profile.baseline
        
        # Calculate total power
        total_power = cpu_power + ram_power + disk_power + gpu_power + baseline_power
        
        # Apply PUE to account for data center overhead
        total_power_with_pue = total_power * self.pue
        
        return {
            'power_watts': total_power,
            'power_with_pue_watts': total_power_with_pue,
            'cpu_power_watts': cpu_power,
            'ram_power_watts': ram_power,
            'disk_power_watts': disk_power,
            'gpu_power_watts': gpu_power,
            'baseline_power_watts': baseline_power
        }
    
    def apply_workload_profile(self, workload_type: str, base_power_watts: float) -> Dict[str, Any]:
        """Apply workload-specific profile to adjust power estimates."""
        profile = AIWorkloadProfiler.get_profile(workload_type)
        
        # Calculate adjustment factor based on the profile
        # This is a weighted average of the different components
        cpu_weight = 0.4
        gpu_weight = 0.4 if self.gpu_enabled else 0
        ram_weight = 0.1
        io_weight = 0.1
        
        # Redistribute weight if GPU is not available
        if not self.gpu_enabled:
            cpu_weight += 0.3
            ram_weight += 0.05
            io_weight += 0.05
        
        # Calculate adjustment factor
        adjustment_factor = (
            (cpu_weight * profile['cpu_multiplier']) +
            (gpu_weight * profile['gpu_multiplier']) +
            (ram_weight * profile['ram_multiplier']) +
            (io_weight * profile['io_multiplier'])
        )
        
        # Apply adjustment
        adjusted_power = base_power_watts * adjustment_factor
        
        return {
            'workload_type': workload_type,
            'workload_description': AIWorkloadProfiler.profile_description(workload_type),
            'power_adjustment_factor': adjustment_factor,
            'adjusted_power_watts': adjusted_power
        }
    
    def estimate_carbon(self, power_watts: float) -> Dict[str, float]:
        """
        Estimate carbon emissions based on power consumption.
        
        Args:
            power_watts: Power consumption in watts
            
        Returns:
            Dictionary with carbon metrics
        """
        # Calculate energy in kWh
        kwh = power_watts / 1000 * self.interval / 3600
        
        # Get regional carbon intensity in gCO2/kWh
        carbon_intensity = self.carbon_intensity.get_intensity()
        
        # Calculate CO2 emissions in grams
        co2_grams = kwh * carbon_intensity
        
        return {
            'energy_kwh': kwh,
            'carbon_intensity': carbon_intensity,
            'co2_grams': co2_grams
        }
    
    def start(self):
        """Start metrics collection."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_metrics)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop metrics collection."""
        if self.running:
            self.running = False
            self.thread.join(timeout=self.interval*2)
    
    def report(self) -> pd.DataFrame:
        """Generate a DataFrame with all collected metrics."""
        if not self.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data)
        
        # Calculate derived metrics
        if len(df) > 1:
            # Calculate rates of change for cumulative metrics
            df['power_rate_watts'] = df['power_watts'].diff() / df['timestamp'].diff().dt.total_seconds()
            df['co2_rate_gps'] = df['co2_grams'].diff() / df['timestamp'].diff().dt.total_seconds()
        
        return df
    
    def summarize(self) -> pd.DataFrame:
        """Generate an overall summary of collected metrics."""
        df = self.report()
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate duration
        if len(df) > 1:
            duration_sec = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        else:
            duration_sec = self.interval
        
        # Prepare summary dictionary
        summary = {
            'Start time': df['timestamp'].min(),
            'End time': df['timestamp'].max(),
            'Duration (s)': duration_sec,
            'CPU avg (%)': df['cpu_percent'].mean(),
            'CPU max (%)': df['cpu_percent'].max(),
            'RAM avg (%)': df['ram_percent'].mean(),
            'RAM max (%)': df['ram_percent'].max(),
            'RAM avg (GB)': df['ram_used_gb'].mean(),
            'Disk avg (%)': df['disk_percent'].mean()
        }
        
        # Add disk I/O if available
        if 'disk_read_mbps' in df.columns:
            summary['Disk read avg (MB/s)'] = df['disk_read_mbps'].mean()
            summary['Disk write avg (MB/s)'] = df['disk_write_mbps'].mean()
        
        # Add GPU metrics if available
        if 'gpu_percent' in df.columns:
            summary['GPU avg (%)'] = df['gpu_percent'].mean()
            summary['GPU max (%)'] = df['gpu_percent'].max()
            summary['GPU memory avg (%)'] = df['gpu_mem_percent'].mean() if 'gpu_mem_percent' in df.columns else 0
        
        # Add network metrics if available
        if 'net_sent_mbps' in df.columns:
            summary['Network sent avg (Mbps)'] = df['net_sent_mbps'].mean()
            summary['Network recv avg (Mbps)'] = df['net_recv_mbps'].mean()
        
        # Add energy and carbon metrics
        power_col = 'adjusted_power_watts' if 'adjusted_power_watts' in df.columns else 'power_watts'
        
        summary['Avg power (W)'] = df[power_col].mean()
        summary['Max power (W)'] = df[power_col].max()
        summary['Total energy (kWh)'] = df['energy_kwh'].sum()
        summary['Total CO2 (g)'] = df['co2_grams'].sum()
        summary['Carbon intensity (gCO2/kWh)'] = df['carbon_intensity'].iloc[0]
        
        # Convert to DataFrame
        summary_df = pd.DataFrame([summary])
        
        # Save summary to JSON if output directory is specified
        if self.output_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(self.output_dir, f"resource_summary_{timestamp}.json")
            
            # Convert timestamps to strings for JSON serialization
            summary_json = summary.copy()
            for key in ['Start time', 'End time']:
                if key in summary_json and isinstance(summary_json[key], datetime.datetime):
                    summary_json[key] = summary_json[key].isoformat()
            
            with open(summary_file, 'w') as f:
                json.dump(summary_json, f, indent=2)
        
        return summary_df
    
    def summarize_sections(self) -> pd.DataFrame:
        """Generate a summary grouped by sections."""
        df = self.report()
        
        if df.empty or 'section' not in df.columns:
            return pd.DataFrame()
        
        # Drop rows with no section
        df_sections = df.dropna(subset=['section'])
        
        if df_sections.empty:
            return pd.DataFrame()
        
        # Select power column
        power_col = 'adjusted_power_watts' if 'adjusted_power_watts' in df.columns else 'power_watts'
        
        # Add GPU metrics if available
        agg_dict = {
            'timestamp': ['min', 'max', lambda x: (x.max() - x.min()).total_seconds() or len(x) * self.interval],
            'cpu_percent': ['mean', 'max'],
            'ram_percent': ['mean', 'max'],
            'ram_used_gb': 'mean',
            'disk_percent': 'mean',
            power_col: ['mean', 'max'],
            'energy_kwh': 'sum',
            'co2_grams': 'sum'
        }
        
        # Add GPU metrics if available
        if 'gpu_percent' in df.columns:
            agg_dict['gpu_percent'] = ['mean', 'max']
            if 'gpu_mem_percent' in df.columns:
                agg_dict['gpu_mem_percent'] = 'mean'
        
        # Add network metrics if available
        if 'net_sent_mbps' in df.columns:
            agg_dict['net_sent_mbps'] = 'mean'
            agg_dict['net_recv_mbps'] = 'mean'
        
        # Add workload type if available
        if 'workload_type' in df.columns:
            agg_dict['workload_type'] = lambda x: x.iloc[0] if not x.isna().all() else None
        
        # Group by section
        agg_mapped = {
            'start_time': ('timestamp', 'min'),
            'end_time': ('timestamp', 'max'),
            'duration_sec': ('timestamp', lambda x: (x.max() - x.min()).total_seconds() or len(x) * self.interval),
            'avg_cpu': ('cpu_percent', 'mean'),
            'max_cpu': ('cpu_percent', 'max'),
            'avg_ram': ('ram_percent', 'mean'),
            'max_ram': ('ram_percent', 'max'),
            'avg_ram_gb': ('ram_used_gb', 'mean'),
            'avg_disk': ('disk_percent', 'mean'),
            'avg_power_watts': (power_col, 'mean'),
            'max_power_watts': (power_col, 'max'),
            'total_energy_kwh': ('energy_kwh', 'sum'),
            'total_co2': ('co2_grams', 'sum')
        }
        
        # Add GPU metrics if available
        if 'gpu_percent' in df.columns:
            agg_mapped['avg_gpu'] = ('gpu_percent', 'mean')
            agg_mapped['max_gpu'] = ('gpu_percent', 'max')
            if 'gpu_mem_percent' in df.columns:
                agg_mapped['avg_gpu_mem'] = ('gpu_mem_percent', 'mean')
        
        # Add network metrics if available
        if 'net_sent_mbps' in df.columns:
            agg_mapped['avg_net_sent'] = ('net_sent_mbps', 'mean')
            agg_mapped['avg_net_recv'] = ('net_recv_mbps', 'mean')
        
        # Add workload type if available
        if 'workload_type' in df.columns:
            agg_mapped['workload_type'] = ('workload_type', lambda x: x.iloc[0] if not x.isna().all() else None)
        
        # Group by section
        section_summary = df_sections.groupby('section').agg(**agg_mapped)
            start_time=('timestamp', 'min'),
            end_time=('timestamp', 'max'),
            duration_sec=('timestamp', lambda x: (x.max() - x.min()).total_seconds() or len(x) * self.interval),
            avg_cpu=('cpu_percent', 'mean'),
            max_cpu=('cpu_percent', 'max'),
            avg_ram=('ram_percent', 'mean'),
            max_ram=('ram_percent', 'max'),
            avg_ram_gb=('ram_used_gb', 'mean'),
            avg_disk=('disk_percent', 'mean'),
            avg_power_watts=(power_col, 'mean'),
            max_power_watts=(power_col, 'max'),
            total_energy_kwh=('energy_kwh', 'sum'),
            total_co2=('co2_grams', 'sum'),
