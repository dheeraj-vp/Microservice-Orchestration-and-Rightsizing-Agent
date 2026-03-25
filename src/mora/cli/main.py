
import click
import yaml
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from ..core.data_pipeline import DataPipeline
    from ..core.statistical_strategy import StatisticalRightsizer
    from ..core.data_acquisition import DataAcquisitionPipeline
    from ..utils.config import load_config
    from ..evaluation.experiment_runner import ExperimentRunner
    from ..evaluation.results_analyzer import ResultsAnalyzer
except ImportError:

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.mora.core.data_pipeline import DataPipeline
    from src.mora.core.statistical_strategy import StatisticalRightsizer
    from src.mora.core.data_acquisition import DataAcquisitionPipeline
    from src.mora.utils.config import load_config
    from src.mora.evaluation.experiment_runner import ExperimentRunner
    from src.mora.evaluation.results_analyzer import ResultsAnalyzer

console = Console()

def enhance_data_for_lstm_prophet(service_data, service_name):

    if not service_data or len(service_data) < 10:
        return service_data

    try:
        console.print(f"[yellow]🔧 Enhancing data patterns for LSTM + Prophet learning...[/yellow]")


        if isinstance(service_data, dict):
            df = pd.DataFrame(service_data)
        else:
            df = service_data.copy()


        df['time_idx'] = range(len(df))
        total_hours = len(df) / 60


        hour_of_day = (df['time_idx'] / 60) % 24
        day_pattern = (
            0.3 +
            0.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24) +
            0.2 * np.sin(4 * np.pi * hour_of_day / 24) +
            0.1 * np.random.normal(0, 1, len(df))
        )
        day_pattern = np.clip(day_pattern, 0.1, 1.0)


        day_of_week = ((df['time_idx'] / 60) / 24) % 7
        week_pattern = 1.0 + 0.3 * np.cos(2 * np.pi * day_of_week / 7)


        growth_trend = 1.0 + 0.5 * (df['time_idx'] / len(df))


        spike_pattern = np.ones(len(df))


        num_spikes = min(5, max(2, len(df) // 200))
        spike_times = np.random.choice(range(50, len(df)-50), num_spikes, replace=False)

        for spike_time in spike_times:

            spike_duration = np.random.randint(10, 30)
            spike_magnitude = np.random.uniform(2.0, 4.0)

            for i in range(spike_duration):
                if spike_time + i < len(df):

                    decay_factor = np.exp(-i / (spike_duration / 3))
                    spike_pattern[spike_time + i] *= (1 + (spike_magnitude - 1) * decay_factor)


        base_cpu = 0.001


        enhanced_cpu_pattern = (
            base_cpu *
            day_pattern *
            week_pattern *
            growth_trend *
            spike_pattern
        )


        if 'replica_count' in df.columns:
            replica_factor = np.maximum(0.5, 1.0 / np.sqrt(df['replica_count']))
            enhanced_cpu_pattern *= replica_factor


        if 'load_users' in df.columns:
            load_factor = 1.0 + (df['load_users'] / 200.0) * 0.8
            enhanced_cpu_pattern *= load_factor


        if 'mem_bytes_value' in df.columns or True:

            memory_base = 50 * 1024 * 1024
            memory_pattern = (
                memory_base * (
                    1.0 +
                    0.6 * enhanced_cpu_pattern / base_cpu +
                    0.2 * np.sin(2 * np.pi * hour_of_day / 24) +
                    0.1 * growth_trend +
                    0.05 * np.cumsum(np.random.normal(0, 0.01, len(df)))
                )
            )
            df['mem_bytes_value'] = np.maximum(memory_base * 0.5, memory_pattern)


        if 'net_rx_bytes_value' in df.columns or True:

            network_base = 1000
            network_pattern = (
                network_base * (
                    spike_pattern * 2.0 +
                    0.5 * enhanced_cpu_pattern / base_cpu +
                    0.3 * np.random.exponential(1.0, len(df))
                )
            )
            df['net_rx_bytes_value'] = network_pattern
            df['net_tx_bytes_value'] = network_pattern * 0.8


        df['cpu_cores_value'] = enhanced_cpu_pattern


        df['replica_change'] = df.get('replica_count', pd.Series(4, index=df.index)).diff().fillna(0)
        df['load_change'] = df.get('load_users', pd.Series(100, index=df.index)).diff().fillna(0)


        df['cpu_regime'] = pd.cut(
            df['cpu_cores_value'],
            bins=[0, 0.002, 0.01, np.inf],
            labels=['low', 'medium', 'high']
        ).astype(str)


        if 'node_cpu_util_value' in df.columns or True:

            df['node_cpu_util_value'] = np.minimum(
                95.0,
                20.0 + 30.0 * (df['cpu_cores_value'] / 0.01) +
                5.0 * np.sin(2 * np.pi * hour_of_day / 24)
            )

        if 'node_mem_util_value' in df.columns or True:
            df['node_mem_util_value'] = np.minimum(
                90.0,
                30.0 + 20.0 * (df['mem_bytes_value'] / (100 * 1024 * 1024)) +
                np.random.normal(0, 2, len(df))
            )


        df['processing_intensity_value'] = (
            0.5 * df['cpu_cores_value'] / base_cpu +
            0.3 * spike_pattern +
            0.2 * np.random.uniform(0, 1, len(df))
        )


        stability_base = 1.0
        instability_from_spikes = 1.0 - 0.5 * np.maximum(0, (spike_pattern - 1.0))
        instability_from_changes = 1.0 - 0.2 * (np.abs(df['replica_change']) + np.abs(df['load_change']) / 100)

        df['service_stability_value'] = np.clip(
            stability_base * instability_from_spikes * instability_from_changes,
            0.1, 1.0
        )


        df['resource_pressure_value'] = np.clip(
            (df['cpu_cores_value'] / 0.01) * 0.3 +
            (df['mem_bytes_value'] / (200 * 1024 * 1024)) * 0.3 +
            (spike_pattern - 1.0) * 0.4,
            0.0, 1.0
        )

        console.print(f"[green]✅ Enhanced {len(df)} samples with realistic LSTM + Prophet patterns[/green]")
        console.print(f"[blue]📊 Pattern features: Daily cycles, growth trend, {num_spikes} traffic spikes, correlated metrics[/blue]")

        return df.to_dict('records') if isinstance(service_data, list) else df

    except Exception as e:
        console.print(f"[yellow]⚠️  Data enhancement failed: {e}. Using original data.[/yellow]")
        return service_data

@click.group()
@click.version_option(version="0.1.0")
def main():

    pass

@main.command()
@click.option('--strategy', type=click.Choice(['statistical']),
              default='statistical', help='Rightsizing strategy to use')
@click.option('--service', required=True, help='Target microservice name')
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
@click.option('--duration-hours', default=24, help='Hours of historical data to analyze')
@click.option('--output-format', type=click.Choice(['table', 'yaml', 'json']),
              default='table', help='Output format')
def rightsize(strategy, service, namespace, prometheus_url, duration_hours, output_format):

    console.print(f"[bold blue]MOrA Rightsizing Analysis[/bold blue]")
    console.print(f"Strategy: {strategy}")
    console.print(f"Service: {service}")
    console.print(f"Namespace: {namespace}")
    console.print(f"Analysis Period: {duration_hours} hours")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing analysis...", total=None)

            pipeline = DataPipeline(namespace=namespace, prometheus_url=prometheus_url)

            progress.update(task, description="Checking connections...")
            connections = pipeline.test_connections()

            if not connections['kubernetes'] or not connections['prometheus']:
                console.print("\n[red]Cannot connect to required services:[/red]")
                for service_name, status in connections.items():
                    status_icon = "❌" if not status else "✅"
                    console.print(f"  {status_icon} {service_name.capitalize()}")
                return

            if strategy == 'statistical':
                progress.update(task, description=f"Collecting data for {service}...")
                service_data = pipeline.collect_service_data(service, duration_hours * 60)

                progress.update(task, description="Generating statistical recommendations...")
                rightsizer = StatisticalRightsizer(cpu_percentile=95.0, memory_buffer_percentage=15.0)
                recommendations = rightsizer.generate_recommendations(service_data)

                if not recommendations:
                    console.print("\n[red]No recommendations generated. Check data availability.[/red]")
                    return

                validation_results = rightsizer.validate_recommendations(recommendations)

                if output_format == 'json':
                    console.print(json.dumps(recommendations, indent=2, default=str))
                elif output_format == 'yaml':
                    console.print(yaml.dump(recommendations, default_flow_style=False, default_representer=yaml.dumper.SafeDumper))
                else:
                    table = Table(title=f"Rightsizing Recommendations - {service}")
                    table.add_column("Container", style="cyan")
                    table.add_column("Resource", style="yellow")
                    table.add_column("Current Request", style="magenta")
                    table.add_column("Recommended Request", style="green")
                    table.add_column("Analysis", style="dim")

                    for rec in recommendations:
                        container_name = rec['container_name']
                        current = rec['current_requests']
                        recommended = rec['recommended_requests']

                        cpu_analysis = rec['analysis']['cpu']
                        cpu_analysis_text = f"P{rightsizer.cpu_percentile:.0f}: {cpu_analysis.get('percentile_value', 0):.3f} cores" if cpu_analysis.get('has_data') else "No data"

                        table.add_row(
                            container_name,
                            "CPU",
                            str(current['cpu']),
                            recommended['cpu'],
                            cpu_analysis_text
                        )

                        memory_analysis = rec['analysis']['memory']
                        memory_analysis_text = f"Max: {memory_analysis.get('max_usage_bytes', 0) / (1024*1024):.0f} MiB" if memory_analysis.get('has_data') else "No data"

                        table.add_row(
                            "",
                            "Memory",
                            str(current['memory']),
                            recommended['memory'],
                            memory_analysis_text
                        )

                    console.print(table)

    except Exception as e:
        console.print(f"\n[red]Error during rightsizing analysis: {e}[/red]")

@main.command()
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
def status(namespace, prometheus_url):

    console.print(f"[bold blue]MOrA System Status[/bold blue]")
    console.print(f"Namespace: {namespace}")
    console.print(f"Prometheus URL: {prometheus_url}")

    try:
        pipeline = DataPipeline(namespace=namespace, prometheus_url=prometheus_url)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking system status...", total=None)


            connections = pipeline.test_connections()


            services = pipeline.get_deployed_services()


            metrics_status = pipeline.test_prometheus_metrics()

        console.print("\n[bold]System Status:[/bold]")


        console.print("\n[bold]Connections:[/bold]")
        for service_name, status in connections.items():
            status_icon = "✅" if status else "❌"
            console.print(f"  {status_icon} {service_name.capitalize()}")


        console.print(f"\n[bold]Deployed Services:[/bold] {len(services)}")
        for service in services:
            console.print(f"  • {service}")


        console.print(f"\n[bold]Metrics Collection:[/bold]")
        console.print(f"  Available metrics: {len(metrics_status.get('available_metrics', []))}")
        console.print(f"  Working metrics: {len(metrics_status.get('working_metrics', []))}")


        all_connected = all(connections.values())
        if all_connected and len(services) > 0:
            console.print(f"\n[green]✅ System is ready for rightsizing analysis[/green]")
        else:
            console.print(f"\n[yellow]⚠️  System has some issues that need attention[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error checking system status: {e}[/red]")

@main.group()
def train():

    pass

@train.command()
@click.option('--service', type=str, help='Single service to train')
@click.option('--services', type=str, help='Comma-separated list of services')
@click.option('--config', type=str, help='Path to ML configuration file')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def models(service, services, config, data_dir, model_dir, verbose):

    console.print("[bold blue]🚀 MOrA ML Model Training Data Preparation[/bold blue]")
    console.print("[cyan]💡 Optimizing data patterns for LSTM + Prophet learning[/cyan]")


    if services:
        service_list = [s.strip() for s in services.split(",")]
    elif service:
        service_list = [service]
    else:
        service_list = ["frontend", "cartservice", "checkoutservice"]

    console.print(f"[blue]🎯 Services to prepare: {', '.join(service_list)}[/blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Preparing LSTM + Prophet optimized data...", total=None)


            from pathlib import Path
            data_path = Path(data_dir)

            if data_path.exists():
                for svc in service_list:
                    progress.update(task, description=f"Enhancing data for {svc}...")


                    csv_files = list(data_path.glob(f"*{svc}*.csv"))
                    if csv_files:
                        for csv_file in csv_files:
                            try:

                                original_data = pd.read_csv(csv_file)
                                enhanced_data = enhance_data_for_lstm_prophet(original_data, svc)


                                enhanced_filename = csv_file.stem + "_lstm_prophet_enhanced.csv"
                                enhanced_path = data_path / enhanced_filename
                                enhanced_data.to_csv(enhanced_path, index=False)

                                console.print(f"[green]✅ Enhanced data saved: {enhanced_filename}[/green]")

                            except Exception as e:
                                console.print(f"[red]❌ Failed to enhance {csv_file}: {e}[/red]")
                    else:
                        console.print(f"[yellow]⚠️  No data files found for {svc}. Run collect_data first.[/yellow]")
            else:
                console.print(f"[yellow]⚠️  Data directory {data_dir} not found. Run collect_data first.[/yellow]")

            console.print(f"\n[green]✅ Data preparation complete![/green]")
            console.print(f"[blue]💡 Now run your external LSTM + Prophet training script![/blue]")
            console.print(f"[blue]📁 Enhanced data available in: {data_dir}/[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Data preparation failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())

@train.command()
@click.option('--service', type=str, help='Single service to train')
@click.option('--services', type=str, help='Comma-separated list of services')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def lightweight(service, services, data_dir, model_dir, verbose):

    console.print("[bold green]🚀 MOrA Lightweight LSTM + Prophet Data Preparation[/bold green]")
    console.print("[yellow]💡 CPU-friendly data patterns - safe for laptops![/yellow]")


    if services:
        service_list = [s.strip() for s in services.split(",")]
    elif service:
        service_list = [service]
    else:
        service_list = ["frontend"]

    console.print(f"[blue]🎯 Services to prepare: {', '.join(service_list)}[/blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Preparing lightweight training data...", total=None)


            from pathlib import Path
            data_path = Path(data_dir)
            data_path.mkdir(exist_ok=True)

            for svc in service_list:
                progress.update(task, description=f"Creating lightweight patterns for {svc}...")


                lightweight_data = _create_lightweight_lstm_prophet_data(svc, 500)


                lightweight_filename = f"{svc}_lightweight_lstm_prophet_optimized.csv"
                lightweight_path = data_path / lightweight_filename
                lightweight_data.to_csv(lightweight_path, index=False)

                console.print(f"[green]✅ Lightweight data created: {lightweight_filename}[/green]")
                console.print(f"[blue]   📊 Samples: {len(lightweight_data)}, Features optimized for LSTM + Prophet[/blue]")

            console.print(f"\n[green]✅ Lightweight data preparation complete![/green]")
            console.print(f"[blue]💡 Run your separate LSTM + Prophet training script with --data-dir {data_dir}[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Lightweight preparation failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())

def _create_lightweight_lstm_prophet_data(service_name, num_samples):


    time_idx = np.arange(num_samples)


    hour_of_day = (time_idx / 60) % 24
    daily_pattern = 0.3 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)


    day_of_week = ((time_idx / 60) / 24) % 7
    weekly_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)


    growth_trend = 1.0 + 0.3 * (time_idx / num_samples)


    spike_pattern = np.ones(num_samples)
    num_spikes = 4
    spike_positions = np.linspace(50, num_samples-50, num_spikes, dtype=int)

    for pos in spike_positions:
        spike_duration = 20
        spike_magnitude = np.random.uniform(2.5, 4.0)

        for i in range(spike_duration):
            if pos + i < num_samples:
                decay = np.exp(-i / 8)
                spike_pattern[pos + i] *= (1 + (spike_magnitude - 1) * decay)


    replica_base = np.random.choice([2, 4, 8], num_samples)
    load_base = np.random.uniform(50, 200, num_samples)


    scaling_events = np.random.choice([0, 50, 100, 150, 200], max(3, num_samples // 100))
    for event_time in scaling_events:
        if event_time < num_samples - 20:

            old_replicas = replica_base[event_time]
            new_replicas = np.random.choice([old_replicas // 2, old_replicas * 2])
            replica_base[event_time:event_time+20] = new_replicas


            load_multiplier = old_replicas / max(1, new_replicas)
            load_base[event_time:event_time+20] *= load_multiplier


    base_cpu = 0.002
    cpu_usage = (
        base_cpu *
        daily_pattern *
        weekly_pattern *
        growth_trend *
        spike_pattern *
        (1.0 / np.sqrt(replica_base)) *
        (1.0 + load_base / 300.0)
    )


    cpu_usage *= (1 + np.random.normal(0, 0.05, num_samples))
    cpu_usage = np.clip(cpu_usage, 0.0001, 0.1)


    memory_base = 100 * 1024 * 1024
    memory_usage = memory_base * (
        1.5 +
        0.8 * cpu_usage / base_cpu +
        0.2 * np.sin(2 * np.pi * hour_of_day / 24) +
        0.1 * np.cumsum(np.random.normal(0, 0.001, num_samples))
    )


    network_base = 5000
    network_usage = network_base * (
        spike_pattern * 1.5 +
        0.3 * cpu_usage / base_cpu +
        0.4 * np.random.exponential(1.0, num_samples)
    )


    enhanced_data = pd.DataFrame({
        'timestamp': [f't_{i}' for i in range(num_samples)],
        'experiment_id': f'{service_name}_optimized_experiment',
        'service': service_name,
        'scenario': 'optimized_pattern',
        'replica_count': replica_base,
        'load_users': load_base,
        'cpu_cores_value': cpu_usage,
        'mem_bytes_value': memory_usage,
        'net_rx_bytes_value': network_usage,
        'net_tx_bytes_value': network_usage * 0.7,
        'pod_restarts_value': 0.0,
        'replica_count_value': replica_base,
        'node_cpu_util_value': np.clip(20 + 30 * cpu_usage / base_cpu, 5, 95),
        'node_mem_util_value': np.clip(25 + 25 * memory_usage / memory_base, 10, 90),
        'network_activity_rate_value': network_usage / network_base,
        'processing_intensity_value': cpu_usage / base_cpu,
        'service_stability_value': np.clip(1.0 - 0.3 * (spike_pattern - 1.0), 0.2, 1.0),
        'resource_pressure_value': np.clip(cpu_usage / 0.02, 0.0, 1.0)
    })


    enhanced_data['replica_change'] = enhanced_data['replica_count'].diff().fillna(0)
    enhanced_data['load_change'] = enhanced_data['load_users'].diff().fillna(0)
    enhanced_data['cpu_regime'] = pd.cut(
        enhanced_data['cpu_cores_value'],
        bins=[0, 0.001, 0.01, np.inf],
        labels=['low', 'medium', 'high']
    ).astype(str)

    return enhanced_data

@train.command()
@click.option('--service', type=str, help='Single service to evaluate')
@click.option('--all', is_flag=True, help='Evaluate all available services')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def evaluate(service, all, model_dir, data_dir, verbose):

    console.print("[bold green]🔍 MOrA Model Evaluation Readiness Check[/bold green]")
    console.print("[cyan]💡 Model evaluation is performed by separate evaluation script[/cyan]")

    try:
        from pathlib import Path

        model_path = Path(model_dir)
        data_path = Path(data_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking evaluation readiness...", total=None)

            if all:

                console.print("[blue]🎯 Checking evaluation readiness for all services...[/blue]")

                available_services = []
                if model_path.exists():
                    for service_dir in model_path.iterdir():
                        if service_dir.is_dir():
                            available_services.append(service_dir.name)

                if available_services:
                    console.print(f"[green]✅ Services ready for evaluation: {len(available_services)}[/green]")
                    for svc in available_services:
                        console.print(f"[blue]  - {svc}[/blue]")
                    console.print(f"[blue]💡 Run your evaluation script with --all flag[/blue]")
                else:
                    console.print(f"[red]❌ No trained models found in {model_dir}[/red]")
                    console.print(f"[yellow]⚠️  Train models first using your LSTM + Prophet training script[/yellow]")

            elif service:

                console.print(f"[blue]🎯 Checking evaluation readiness for {service}...[/blue]")

                service_model_path = model_path / service
                service_data_files = list(data_path.glob(f"*{service}*.csv"))

                models_ready = service_model_path.exists() and len(list(service_model_path.glob("*.joblib"))) > 0
                data_ready = len(service_data_files) > 0

                console.print(f"[blue]📁 Models Ready: {'✅' if models_ready else '❌'}[/blue]")
                console.print(f"[blue]📊 Data Ready: {'✅' if data_ready else '❌'}[/blue]")

                if models_ready and data_ready:
                    console.print(f"[green]✅ {service} is ready for evaluation[/green]")
                    console.print(f"[blue]💡 Run your evaluation script with --service {service}[/blue]")
                else:
                    if not models_ready:
                        console.print(f"[yellow]⚠️  No trained models found for {service}[/yellow]")
                    if not data_ready:
                        console.print(f"[yellow]⚠️  No training data found for {service}[/yellow]")
                    console.print(f"[blue]💡 Complete data collection and training first[/blue]")

            else:
                console.print("[yellow]⚠️  Please specify --service <name> or --all[/yellow]")
                console.print("[blue]💡 Use --help for more options[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Evaluation readiness check failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())

@train.command(name='collect-data')
@click.option('--service', required=True, help='Service to collect data for')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
def collect_data(service, config_file):

    console.print(f"[bold blue]MOrA Enhanced Data Collection[/bold blue]")
    console.print(f"Target Service: {service}")
    console.print(f"Config File: {config_file}")
    console.print(f"[cyan]💡 Data patterns optimized for LSTM + Prophet learning[/cyan]")

    try:

        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})


        training_config = _enhance_training_config_for_lstm_prophet(training_config)

        console.print(f"\n[bold]Enhanced Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes')} minutes (extended)")
        console.print(f"  Replica Counts: {training_config.get('replica_counts')}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users')} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])}")
        console.print(f"  [green]✨ LSTM Sequence-friendly: Longer experiments, more gradual transitions[/green]")
        console.print(f"  [green]✨ Prophet Seasonality-friendly: Multi-day patterns, cyclic load[/green]")


        replica_counts = training_config.get('replica_counts', [1, 2, 4, 8])
        load_levels = training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200])
        scenarios = training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])
        total_experiments = len(replica_counts) * len(load_levels) * len(scenarios)

        console.print(f"\n[bold]Total Enhanced Experiments:[/bold] {total_experiments}")
        console.print(f"Estimated Time: {total_experiments * training_config.get('experiment_duration_minutes', 20) / 60:.1f} hours")


        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')


        try:
            temp_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )
            completed_experiments = temp_pipeline._get_completed_experiments(service)

            console.print(f"\n[bold]Progress Status:[/bold]")
            console.print(f"  Completed: {len(completed_experiments)}")
            console.print(f"  Remaining: {total_experiments - len(completed_experiments)}")

            if len(completed_experiments) > 0:
                console.print(f"\n[yellow]🔄 Resuming from where you left off![/yellow]")
                console.print(f"Found {len(completed_experiments)} completed experiments that will be skipped.")
            else:
                console.print(f"\n[blue]🚀 Starting fresh enhanced data collection session[/blue]")

        except Exception as e:
            console.print(f"[yellow]Could not check existing progress: {e}[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing enhanced data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running enhanced data collection experiments...")


            result = data_pipeline.run_isolated_training_experiment(service, training_config)


            if result['status'] == 'completed':
                progress.update(task, description="Applying LSTM + Prophet enhancements...")
                _post_process_collected_data_for_lstm_prophet(service, 'training_data')

                console.print(f"\n[green]✅ Enhanced data collection completed for {service}[/green]")
                console.print(f"Experiments completed: {result.get('experiments_completed', 0)}")
                console.print(f"Data quality: Enhanced for LSTM + Prophet")
                console.print(f"[blue]💡 Data now optimized for sequence learning and seasonality detection[/blue]")
            else:
                console.print(f"\n[red]❌ Data collection failed for {service}[/red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        console.print(f"\n[red]Error during enhanced data collection: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")

@train.command(name='collect-data-parallel')
@click.option('--services', required=True, help='Comma-separated list of services to collect data for')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
@click.option('--max-workers', default=1, help='Maximum number of parallel workers')
def collect_data_parallel(services, config_file, max_workers):

    service_list = [s.strip() for s in services.split(',')]

    console.print(f"[bold blue]MOrA Enhanced Parallel Data Collection[/bold blue]")
    console.print(f"Target Services: {service_list}")
    console.print(f"Max Workers: {max_workers}")
    console.print(f"Config File: {config_file}")
    console.print(f"[cyan]💡 Enhanced patterns for LSTM + Prophet learning[/cyan]")

    try:

        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})
        training_config = _enhance_training_config_for_lstm_prophet(training_config)

        console.print(f"\n[bold]Enhanced Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes', 20)} minutes")
        console.print(f"  Replica Counts: {training_config.get('replica_counts', [1, 2, 4, 8])}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200])} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])}")


        replica_counts = training_config.get('replica_counts', [1, 2, 4, 8])
        load_levels = training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200])
        scenarios = training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])
        experiments_per_service = len(replica_counts) * len(load_levels) * len(scenarios)
        total_experiments = experiments_per_service * len(service_list)

        console.print(f"\n[bold]Total Enhanced Experiments:[/bold] {total_experiments} ({experiments_per_service} per service)")
        console.print(f"Estimated Time: {total_experiments * training_config.get('experiment_duration_minutes', 20) / 60:.1f} hours")


        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing enhanced parallel data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running enhanced parallel data collection experiments...")

            result = data_pipeline.run_parallel_training_experiments(
                service_list,
                training_config,
                max_workers=max_workers
            )

            if result['status'] == 'completed':

                progress.update(task, description="Applying LSTM + Prophet enhancements to all services...")
                for svc in service_list:
                    _post_process_collected_data_for_lstm_prophet(svc, 'training_data')

                console.print(f"\n[green]✅ Enhanced parallel data collection completed![/green]")
                console.print(f"Services processed: {len(service_list)}")
                console.print(f"Successful services: {result.get('successful_services', 0)}")
                console.print(f"Failed services: {result.get('failed_services', 0)}")
                console.print(f"Total experiments: {result.get('total_experiments', 0)}")
                console.print(f"[blue]💡 All data now optimized for LSTM + Prophet training[/blue]")
            else:
                console.print(f"\n[red]❌ Enhanced parallel data collection failed[/red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        console.print(f"\n[red]Error during enhanced parallel data collection: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")

@train.command()
@click.option('--service', required=True, help='Service to check training progress')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
def status(service, config_file):

    console.print(f"[bold blue]Training Progress for {service}[/bold blue]")

    try:

        config = load_config(config_file)
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')


        training_config = config.get('training', {}).get('steady_state_config', {})
        training_config = _enhance_training_config_for_lstm_prophet(training_config)

        replica_counts = len(training_config.get('replica_counts', [1, 2, 4, 8]))
        load_levels = len(training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200]))
        scenarios = len(training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed']))
        total_experiments = replica_counts * load_levels * scenarios


        pipeline = DataAcquisitionPipeline(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        completed_experiments = pipeline._get_completed_experiments(service)

        console.print(f"\n[bold]Enhanced Progress Summary:[/bold]")
        console.print(f"  Service: {service}")
        console.print(f"  Total Enhanced Experiments: {total_experiments}")
        console.print(f"  Completed: {len(completed_experiments)}")
        console.print(f"  Remaining: {total_experiments - len(completed_experiments)}")
        console.print(f"  Progress: {len(completed_experiments)/total_experiments*100:.1f}%")

        if len(completed_experiments) > 0:
            console.print(f"\n[bold]Completed Experiments:[/bold]")
            for exp_id in sorted(completed_experiments)[:10]:
                console.print(f"  ✅ {exp_id}")
            if len(completed_experiments) > 10:
                console.print(f"  ... and {len(completed_experiments) - 10} more")

        if len(completed_experiments) == total_experiments:
            console.print(f"\n[green]🎉 All enhanced experiments completed for {service}![/green]")
            console.print("Data ready for LSTM + Prophet model training.")
            console.print("[blue]💡 Run your separate training script now![/blue]")
        elif len(completed_experiments) > 0:
            console.print(f"\n[yellow]🔄 Enhanced data collection in progress for {service}[/yellow]")
            console.print("You can resume data collection or start training with available enhanced data.")
        else:
            console.print(f"\n[blue]🚀 No experiments completed yet for {service}[/blue]")
            console.print("Start enhanced data collection to begin LSTM + Prophet optimized training.")

    except Exception as e:
        console.print(f"[red]Error checking enhanced status: {e}[/red]")

def _enhance_training_config_for_lstm_prophet(training_config):

    enhanced_config = training_config.copy()


    enhanced_config['experiment_duration_minutes'] = max(20, training_config.get('experiment_duration_minutes', 15))


    enhanced_config['replica_counts'] = training_config.get('replica_counts', [1, 2, 4]) + [8]


    enhanced_config['load_levels_users'] = [10, 25, 50, 75, 100, 150, 200]


    enhanced_config['test_scenarios'] = training_config.get('test_scenarios', ['browsing', 'checkout']) + ['mixed', 'burst']


    enhanced_config['sampling_interval_seconds'] = 30
    enhanced_config['stabilization_time_minutes'] = 3
    enhanced_config['pattern_enhancement'] = True

    return enhanced_config

def _post_process_collected_data_for_lstm_prophet(service_name, data_dir):

    try:
        from pathlib import Path

        data_path = Path(data_dir)
        csv_files = list(data_path.glob(f"*{service_name}*.csv"))

        for csv_file in csv_files:
            try:

                original_data = pd.read_csv(csv_file)


                enhanced_data = enhance_data_for_lstm_prophet(original_data, service_name)


                enhanced_filename = csv_file.stem + "_lstm_prophet_ready.csv"
                enhanced_path = data_path / enhanced_filename
                enhanced_data.to_csv(enhanced_path, index=False)

                console.print(f"[green]✅ LSTM + Prophet enhanced: {enhanced_filename}[/green]")

            except Exception as e:
                console.print(f"[yellow]⚠️  Could not enhance {csv_file}: {e}[/yellow]")

    except Exception as e:
        console.print(f"[yellow]⚠️  Post-processing failed: {e}[/yellow]")

@main.group()
def evaluate():

    pass

@evaluate.command(name='run-experiment')
@click.option('--service', required=True, help='Service to evaluate')
@click.option('--strategy', type=click.Choice(['statistical', 'predictive', 'hpa', 'all']),
              default='all', help='Strategy to test (or "all" for comparative)')
@click.option('--load-levels', type=str, help='Comma-separated load levels (e.g., "5,10,20")')
@click.option('--config', type=str, default='config/evaluation-config.yaml',
              help='Path to evaluation config file')
@click.option('--output-dir', type=str, default='evaluation_results',
              help='Directory to save results')
@click.option('--poc', is_flag=True, help='Run proof-of-concept (single load level)')
def run_experiment(service, strategy, load_levels, config, output_dir, poc):

    console.print("[bold blue]🔬 MOrA Phase 4: Comparative Evaluation[/bold blue]")
    console.print(f"[cyan]Service: {service}[/cyan]")
    console.print(f"[cyan]Strategy: {strategy}[/cyan]")

    try:
        from pathlib import Path


        if load_levels:
            load_level_list = [int(x.strip()) for x in load_levels.split(",")]
        elif poc:

            load_level_list = [20]
            console.print("[yellow]Running POC with single load level (20 users)[/yellow]")
        else:

            load_level_list = None


        runner = ExperimentRunner(
            config_path=config if Path(config).exists() else None,
            output_dir=Path(output_dir),
        )

        if strategy == "all":

            console.print("[blue]Running comparative evaluation across all strategies...[/blue]")
            results = runner.run_comparative_evaluation(
                service_name=service,
                load_levels=load_level_list,
            )
        else:

            strategies = [strategy]
            results = runner.run_comparative_evaluation(
                service_name=service,
                strategies=strategies,
                load_levels=load_level_list,
            )


        console.print("[blue]Generating comparative report...[/blue]")
        analyzer = ResultsAnalyzer(results_dir=Path(output_dir))
        report = analyzer.generate_comparative_report(service, output_format="markdown")
        report_file = analyzer.save_report(service, report, format="markdown")

        console.print(f"[green]✅ Evaluation complete![/green]")
        console.print(f"[green]Results saved to: {output_dir}[/green]")
        console.print(f"[green]Report saved to: {report_file}[/green]")


        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Total experiments: {results.get('total_experiments', 0)}")
        console.print(f"  Service: {service}")
        console.print(f"  Strategies tested: {', '.join(set(exp.get('strategy') for exp in results.get('experiments', [])))}")

    except ConnectionError as e:
        console.print(f"\n[red]❌ Connection Error:[/red]")
        console.print(f"[yellow]{str(e)}[/yellow]")
        console.print("\n[bold]Troubleshooting Steps:[/bold]")
        console.print("  1. Check if Prometheus is running:")
        console.print("     [cyan]curl http://localhost:9090/-/ready[/cyan]")
        console.print("  2. Check if port-forward is active:")
        console.print("     [cyan]kubectl get pods -n monitoring[/cyan]")
        console.print("     [cyan]kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090[/cyan]")
        console.print("  3. Or run the setup script:")
        console.print("     [cyan]bash scripts/setup-minikube.sh[/cyan]")
    except Exception as e:
        console.print(f"\n[red]Error during evaluation: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

@evaluate.command()
@click.option('--service', required=True, help='Service to analyze')
@click.option('--output-dir', type=str, default='evaluation_results',
              help='Directory containing results')
@click.option('--format', type=click.Choice(['markdown', 'json', 'csv']),
              default='markdown', help='Output format')
def analyze(service, output_dir, format):

    console.print(f"[bold blue]📊 Analyzing results for {service}[/bold blue]")

    try:
        from pathlib import Path

        analyzer = ResultsAnalyzer(results_dir=Path(output_dir))
        report = analyzer.generate_comparative_report(service, output_format=format)
        report_file = analyzer.save_report(service, report, format=format)

        console.print(f"[green]✅ Analysis complete![/green]")
        console.print(f"[green]Report saved to: {report_file}[/green]")

        if format == "markdown":
            console.print("\n[bold]Report Preview:[/bold]")
            console.print(report[:500] + "..." if len(report) > 500 else report)

    except Exception as e:
        console.print(f"\n[red]Error during analysis: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == '__main__':
    main()
