"""
Main CLI entry point for MOrA
"""
import click
import yaml
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime

try:
    from ..core.data_pipeline import DataPipeline
    from ..core.statistical_strategy import StatisticalRightsizer
    from ..core.data_acquisition import DataAcquisitionPipeline
    from ..utils.config import load_config
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.mora.core.data_pipeline import DataPipeline
    from src.mora.core.statistical_strategy import StatisticalRightsizer
    from src.mora.core.data_acquisition import DataAcquisitionPipeline
    from src.mora.utils.config import load_config

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """MOrA - Microservices-Aware Orchestrator Agent for Predictive Kubernetes Rightsizing"""
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
    """
    Generate rightsizing recommendations for a microservice
    """
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
    """
    Show current status of monitoring stack and services
    """
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

            # Test connections
            connections = pipeline.test_connections()
            
            # Get service status
            services = pipeline.get_deployed_services()
            
            # Get Prometheus metrics status
            metrics_status = pipeline.test_prometheus_metrics()

        console.print("\n[bold]System Status:[/bold]")
        
        # Connection status
        console.print("\n[bold]Connections:[/bold]")
        for service_name, status in connections.items():
            status_icon = "✅" if status else "❌"
            console.print(f"  {status_icon} {service_name.capitalize()}")

        # Services status
        console.print(f"\n[bold]Deployed Services:[/bold] {len(services)}")
        for service in services:
            console.print(f"  • {service}")

        # Metrics status
        console.print(f"\n[bold]Metrics Collection:[/bold]")
        console.print(f"  Available metrics: {len(metrics_status.get('available_metrics', []))}")
        console.print(f"  Working metrics: {len(metrics_status.get('working_metrics', []))}")

        # Overall status
        all_connected = all(connections.values())
        if all_connected and len(services) > 0:
            console.print(f"\n[green]✅ System is ready for rightsizing analysis[/green]")
        else:
            console.print(f"\n[yellow]⚠️  System has some issues that need attention[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error checking system status: {e}[/red]")


@main.group()
def train():
    """ML model training and data collection commands"""
    pass


@train.command()
@click.option('--service', type=str, help='Single service to train')
@click.option('--services', type=str, help='Comma-separated list of services')
@click.option('--config', type=str, help='Path to ML configuration file')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def models(service, services, config, data_dir, model_dir, verbose):
    """
    Train ML models using advanced algorithms for microservice rightsizing.
    
    This command uses LSTM, Prophet, XGBoost, LightGBM, RandomForest, and GradientBoosting
    algorithms to provide intelligent resource recommendations for CPU, Memory, and Replica scaling.
    """
    console.print("[bold blue]🚀 MOrA ML Model Training[/bold blue]")
    
    # Import ML components
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from train_models.train_professional_ml_pipeline import ProfessionalMLPipeline
    except ImportError as e:
        console.print(f"[red]❌ Error importing ML components: {e}[/red]")
        console.print("Make sure the ML pipeline is properly installed")
        return
    
    # Parse services
    if services:
        service_list = [s.strip() for s in services.split(",")]
    elif service:
        service_list = [service]
    else:
        service_list = ["frontend", "cartservice", "checkoutservice"]
    
    console.print(f"[blue]🎯 Services to train: {', '.join(service_list)}[/blue]")
    
    # Load configuration
    config_data = None
    if config:
        try:
            with open(config, 'r') as f:
                if config.endswith('.json'):
                    config_data = json.load(f)
                else:
                    console.print(f"[yellow]⚠️  Unsupported configuration format: {config}[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error loading configuration: {e}[/red]")
            return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing ML pipeline...", total=None)
            
            # Initialize pipeline
            pipeline = ProfessionalMLPipeline(
                data_dir=data_dir,
                model_dir=model_dir,
                config=config_data
            )
            
            progress.update(task, description="Starting training...")
            
            # Train models
            if len(service_list) == 1:
                console.print(f"[blue]🎯 Training single service: {service_list[0]}[/blue]")
                result = pipeline.train_service(service_list[0])
                
                if result["status"] == "success":
                    console.print(f"[green]✅ Training completed successfully![/green]")
                    console.print(f"[blue]🎯 Service: {result['service_name']}[/blue]")
                    console.print(f"[blue]⏱️  Training Time: {result['training_time']:.2f} seconds[/blue]")
                    console.print(f"[blue]🤖 Models Trained: {result['models_trained']}[/blue]")
                    console.print(f"[blue]🏆 Best Model: {result['best_model']}[/blue]")
                    
                    if "performance" in result:
                        perf = result["performance"]
                        console.print(f"[blue]📊 Performance:[/blue]")
                        console.print(f"[blue]  - R² Score: {perf.get('r2', 'N/A'):.4f}[/blue]")
                        console.print(f"[blue]  - MAE: {perf.get('mae', 'N/A'):.4f}[/blue]")
                        console.print(f"[blue]  - MAPE: {perf.get('mape', 'N/A'):.4f}[/blue]")
                else:
                    console.print(f"[red]❌ Training failed: {result.get('error', 'Unknown error')}[/red]")
            else:
                console.print(f"[blue]🎯 Training multiple services: {', '.join(service_list)}[/blue]")
                results = pipeline.train_all_services(service_list)
                
                console.print(f"[blue]📊 Training Summary:[/blue]")
                console.print(f"[blue]📈 Total Services: {results['total_services']}[/blue]")
                console.print(f"[blue]✅ Successful: {results['successful_services']}[/blue]")
                console.print(f"[blue]❌ Failed: {results['failed_services']}[/blue]")
                console.print(f"[blue]📊 Success Rate: {results['success_rate']:.2%}[/blue]")
                
                if results['successful_services_list']:
                    console.print(f"[green]✅ Successful Services:[/green]")
                    for svc in results['successful_services_list']:
                        console.print(f"[green]  - {svc}[/green]")
                
                if results['failed_services_list']:
                    console.print(f"[red]❌ Failed Services:[/red]")
                    for svc in results['failed_services_list']:
                        console.print(f"[red]  - {svc}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
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
    """
    Train lightweight LSTM + Prophet models (CPU-friendly, fast training).
    
    This command uses only LSTM and Prophet algorithms for efficient, CPU-friendly
    training that won't overheat your system. Perfect for laptops and development.
    """
    console.print("[bold green]🚀 MOrA Lightweight LSTM + Prophet Training[/bold green]")
    console.print("[yellow]💡 CPU-friendly training - safe for laptops![/yellow]")
    
    # Import lightweight ML components
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from train_models.train_lightweight_lstm_prophet import LightweightLSTMProphetPipeline
    except ImportError as e:
        console.print(f"[red]❌ Error importing lightweight ML components: {e}[/red]")
        console.print("Make sure the lightweight ML pipeline is properly installed")
        return
    
    # Parse services
    if services:
        service_list = [s.strip() for s in services.split(",")]
    elif service:
        service_list = [service]
    else:
        service_list = ["frontend"]
    
    console.print(f"[blue]🎯 Services to train: {', '.join(service_list)}[/blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing lightweight pipeline...", total=None)
            
            # Initialize pipeline
            pipeline = LightweightLSTMProphetPipeline(
                data_dir=data_dir,
                model_dir=model_dir
            )
            
            progress.update(task, description="Starting lightweight training...")
            
            # Train models
            if len(service_list) == 1:
                console.print(f"[blue]🎯 Training single service: {service_list[0]}[/blue]")
                result = pipeline.train_pipeline(service_list[0])
                
                if result["status"] == "success":
                    console.print(f"[green]✅ Lightweight training completed successfully![/green]")
                    console.print(f"[blue]🎯 Service: {result['service_name']}[/blue]")
                    console.print(f"[blue]⏱️  Training Time: {result['training_time']:.2f} seconds[/blue]")
                    console.print(f"[blue]📁 Model saved to: {result['model_path']}[/blue]")
                    
                    # Show fusion results
                    if "fusion_results" in result:
                        console.print(f"[blue]🔗 Fusion Results:[/blue]")
                        for target, fusion in result["fusion_results"].items():
                            if fusion["status"] == "success":
                                console.print(f"[blue]  - {target}: {fusion['prediction']:.6f} (confidence: {fusion['confidence']:.2f})[/blue]")
                else:
                    console.print(f"[red]❌ Training failed: {result.get('error', 'Unknown error')}[/red]")
            else:
                console.print(f"[blue]🎯 Training multiple services: {', '.join(service_list)}[/blue]")
                successful_services = []
                failed_services = []
                
                for svc in service_list:
                    console.print(f"[blue]🎯 Training {svc}...[/blue]")
                    result = pipeline.train_pipeline(svc)
                    
                    if result["status"] == "success":
                        successful_services.append(svc)
                        console.print(f"[green]✅ {svc} trained successfully ({result['training_time']:.2f}s)[/green]")
                    else:
                        failed_services.append(svc)
                        console.print(f"[red]❌ {svc} failed: {result.get('error', 'Unknown error')}[/red]")
                
                console.print(f"[blue]📊 Training Summary:[/blue]")
                console.print(f"[blue]📈 Total Services: {len(service_list)}[/blue]")
                console.print(f"[blue]✅ Successful: {len(successful_services)}[/blue]")
                console.print(f"[blue]❌ Failed: {len(failed_services)}[/blue]")
                
                if successful_services:
                    console.print(f"[green]✅ Successful Services:[/green]")
                    for svc in successful_services:
                        console.print(f"[green]  - {svc}[/green]")
                
                if failed_services:
                    console.print(f"[red]❌ Failed Services:[/red]")
                    for svc in failed_services:
                        console.print(f"[red]  - {svc}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Lightweight training failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


@train.command()
@click.option('--service', type=str, help='Single service to evaluate')
@click.option('--all', is_flag=True, help='Evaluate all available services')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def evaluate(service, all, model_dir, data_dir, verbose):
    """
    Evaluate trained models using unified evaluation system.
    
    This command provides comprehensive analysis, industry-standard metrics,
    and production readiness assessment for any trained model.
    """
    console.print("[bold green]🔍 MOrA Unified Model Evaluation[/bold green]")
    
    # Import unified evaluation components
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from evaluate_models.unified_model_evaluator import UnifiedModelEvaluator
    except ImportError as e:
        console.print(f"[red]❌ Error importing evaluation components: {e}[/red]")
        console.print("Make sure the unified evaluation suite is properly installed")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing unified evaluation suite...", total=None)
            
            # Initialize unified evaluator
            evaluator = UnifiedModelEvaluator(
                models_dir=model_dir,
                data_dir=data_dir
            )
            
            progress.update(task, description="Starting evaluation...")
            
            if all:
                # Evaluate all services
                console.print("[blue]🎯 Evaluating all available services...[/blue]")
                result = evaluator.evaluate_all_services()
                
                if result and result['evaluations']:
                    console.print(f"[green]✅ Evaluation completed successfully![/green]")
                    console.print(f"[blue]📊 Services Evaluated: {len(result['evaluations'])}[/blue]")
                    console.print(f"[blue]📈 Average Score: {result['summary_stats']['average_score']:.1f}%[/blue]")
                    console.print(f"[blue]📁 Summary report saved to: {result['summary_path']}[/blue]")
                    
                    # Show individual results
                    console.print(f"[blue]📋 Individual Results:[/blue]")
                    for svc_name, evaluation in result['evaluations'].items():
                        score = evaluation['overall_score']
                        status = "🎉 EXCELLENT" if score >= 80 else "✅ GOOD" if score >= 60 else "⚠️ ACCEPTABLE" if score >= 40 else "❌ BELOW STANDARDS"
                        console.print(f"[blue]  - {svc_name}: {score:.1f}% - {status}[/blue]")
                else:
                    console.print(f"[red]❌ No services could be evaluated[/red]")
            
            elif service:
                # Evaluate single service
                console.print(f"[blue]🎯 Evaluating single service: {service}[/blue]")
                result = evaluator.evaluate_single_service(service)
                
                if result:
                    console.print(f"[green]✅ Evaluation completed successfully![/green]")
                    console.print(f"[blue]🎯 Service: {service}[/blue]")
                    console.print(f"[blue]📊 Overall Score: {result['evaluation']['overall_score']:.1f}%[/blue]")
                    console.print(f"[blue]📁 Report saved to: {result['report_path']}[/blue]")
                    
                    # Show key metrics
                    console.print(f"[blue]📋 Key Metrics:[/blue]")
                    for target, metrics in result['evaluation']['performance_metrics'].items():
                        if 'confidence' in metrics:
                            console.print(f"[blue]  - {target}: Confidence {metrics['confidence']:.2f}[/blue]")
                else:
                    console.print(f"[red]❌ Failed to evaluate service: {service}[/red]")
            
            else:
                console.print("[yellow]⚠️  Please specify --service <name> or --all[/yellow]")
                console.print("[blue]💡 Use --help for more options[/blue]")
    
    except Exception as e:
        console.print(f"[red]❌ Evaluation failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


@train.command(name='collect-data')
@click.option('--service', required=True, help='Service to collect data for')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
def collect_data(service, config_file):
    """
    Collect training data for ML model training.
    
    Runs controlled experiments with different replica counts and load levels
    to collect clean training data for microservice rightsizing models.
    """
    console.print(f"[bold blue]MOrA Data Collection[/bold blue]")
    console.print(f"Target Service: {service}")
    console.print(f"Config File: {config_file}")
    
    try:
        # Load training configuration from config file
        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})
        
        console.print(f"\n[bold]Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes', 15)} minutes")
        console.print(f"  Replica Counts: {training_config.get('replica_counts', [1, 2, 4])}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users', [5, 10, 20, 30, 50, 75])} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing', 'checkout'])}")
        
        # Calculate total experiments
        replica_counts = training_config.get('replica_counts', [1, 2, 4])
        load_levels = training_config.get('load_levels_users', [5, 10, 20, 30, 50, 75])
        scenarios = training_config.get('test_scenarios', ['browsing', 'checkout'])
        total_experiments = len(replica_counts) * len(load_levels) * len(scenarios)
        
        console.print(f"\n[bold]Total Experiments:[/bold] {total_experiments}")
        console.print(f"Estimated Time: {total_experiments * training_config.get('experiment_duration_minutes', 15) / 60:.1f} hours")
        
        # Load namespace and prometheus URL from config
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Check existing progress
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
                console.print(f"\n[blue]🚀 Starting fresh data collection session[/blue]")
            
        except Exception as e:
            console.print(f"[yellow]Could not check existing progress: {e}[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running data collection experiments...")
            
            result = data_pipeline.run_isolated_training_experiment(service, training_config)
            
            if result['status'] == 'completed':
                console.print(f"\n[green]✅ Data collection completed for {service}[/green]")
                console.print(f"Experiments completed: {result.get('experiments_completed', 0)}")
                console.print(f"Data quality: {result.get('data_quality', 'Good')}")
            else:
                console.print(f"\n[red]❌ Data collection failed for {service}[/red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        console.print(f"\n[red]Error during data collection: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")


@train.command(name='collect-data-parallel')
@click.option('--services', required=True, help='Comma-separated list of services to collect data for')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
@click.option('--max-workers', default=1, help='Maximum number of parallel workers')
def collect_data_parallel(services, config_file, max_workers):
    """
    Collect training data for multiple services in parallel.
    
    Runs controlled experiments across multiple services simultaneously
    to dramatically reduce total data collection time.
    """
    service_list = [s.strip() for s in services.split(',')]
    
    console.print(f"[bold blue]MOrA Parallel Data Collection[/bold blue]")
    console.print(f"Target Services: {service_list}")
    console.print(f"Max Workers: {max_workers}")
    console.print(f"Config File: {config_file}")
    
    try:
        # Load training configuration from config file
        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})
        
        console.print(f"\n[bold]Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes', 15)} minutes")
        console.print(f"  Replica Counts: {training_config.get('replica_counts', [1, 2, 4])}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users', [5, 10, 20, 30, 50, 75])} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing', 'checkout'])}")
        
        # Calculate total experiments per service
        replica_counts = training_config.get('replica_counts', [1, 2, 4])
        load_levels = training_config.get('load_levels_users', [5, 10, 20, 30, 50, 75])
        scenarios = training_config.get('test_scenarios', ['browsing', 'checkout'])
        experiments_per_service = len(replica_counts) * len(load_levels) * len(scenarios)
        total_experiments = experiments_per_service * len(service_list)
        
        console.print(f"\n[bold]Total Experiments:[/bold] {total_experiments} ({experiments_per_service} per service)")
        console.print(f"Estimated Time: {total_experiments * training_config.get('experiment_duration_minutes', 15) / 60:.1f} hours")
        
        # Load namespace and prometheus URL from config
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing parallel data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running parallel data collection experiments...")
            
            result = data_pipeline.run_parallel_training_experiments(
                service_list, 
                training_config, 
                max_workers=max_workers
            )
            
            if result['status'] == 'completed':
                console.print(f"\n[green]✅ Parallel data collection completed![/green]")
                console.print(f"Services processed: {len(service_list)}")
                console.print(f"Successful services: {result.get('successful_services', 0)}")
                console.print(f"Failed services: {result.get('failed_services', 0)}")
                console.print(f"Total experiments: {result.get('total_experiments', 0)}")
            else:
                console.print(f"\n[red]❌ Parallel data collection failed[/red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        console.print(f"\n[red]Error during parallel data collection: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")


@train.command()
@click.option('--service', required=True, help='Service to check training progress')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
def status(service, config_file):
    """
    Check training experiment progress for a service.
    """
    console.print(f"[bold blue]Training Progress for {service}[/bold blue]")
    
    try:
        # Load configuration
        config = load_config(config_file)
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Get training config for experiment count
        training_config = config.get('training', {}).get('steady_state_config', {})
        replica_counts = len(training_config.get('replica_counts', [1, 2, 4]))
        load_levels = len(training_config.get('load_levels_users', [5, 10, 20, 30, 50, 75]))
        scenarios = len(training_config.get('test_scenarios', ['browsing', 'checkout']))
        total_experiments = replica_counts * load_levels * scenarios
        
        # Initialize pipeline and check progress
        pipeline = DataAcquisitionPipeline(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        completed_experiments = pipeline._get_completed_experiments(service)
        
        console.print(f"\n[bold]Progress Summary:[/bold]")
        console.print(f"  Service: {service}")
        console.print(f"  Total Experiments: {total_experiments}")
        console.print(f"  Completed: {len(completed_experiments)}")
        console.print(f"  Remaining: {total_experiments - len(completed_experiments)}")
        console.print(f"  Progress: {len(completed_experiments)/total_experiments*100:.1f}%")
        
        if len(completed_experiments) > 0:
            console.print(f"\n[bold]Completed Experiments:[/bold]")
            for exp_id in sorted(completed_experiments)[:10]:  # Show first 10
                console.print(f"  ✅ {exp_id}")
            if len(completed_experiments) > 10:
                console.print(f"  ... and {len(completed_experiments) - 10} more")
        
        if len(completed_experiments) == total_experiments:
            console.print(f"\n[green]🎉 All experiments completed for {service}![/green]")
            console.print("Ready for model training.")
        elif len(completed_experiments) > 0:
            console.print(f"\n[yellow]🔄 Data collection in progress for {service}[/yellow]")
            console.print("You can resume data collection or start training with available data.")
        else:
            console.print(f"\n[blue]🚀 No experiments completed yet for {service}[/blue]")
            console.print("Start data collection to begin training.")
            
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")


if __name__ == '__main__':
    main()
