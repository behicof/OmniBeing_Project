"""
Continuous Integration/Continuous Deployment Pipeline for OmniBeing Trading System.
Automated testing, security scanning, deployment, and rollback capabilities.
"""

import os
import subprocess
import asyncio
import yaml
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import git
import docker
import requests
from production_config import ProductionConfig

@dataclass
class PipelineStage:
    """CI/CD pipeline stage definition."""
    name: str
    commands: List[str]
    timeout: int
    allow_failure: bool = False
    retry_count: int = 0
    dependencies: List[str] = None

@dataclass
class PipelineRun:
    """Pipeline execution run details."""
    run_id: str
    branch: str
    commit_sha: str
    started_at: datetime
    status: str  # 'running', 'success', 'failed', 'cancelled'
    stages: Dict[str, Dict]
    completed_at: Optional[datetime] = None
    logs: List[str] = None

@dataclass
class TestResult:
    """Test execution result."""
    test_suite: str
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage: float
    detailed_results: Dict[str, Any]

@dataclass
class SecurityScanResult:
    """Security scan result."""
    scan_type: str
    vulnerabilities_found: int
    critical: int
    high: int
    medium: int
    low: int
    scan_duration: float
    report_path: str

class ContinuousIntegration:
    """
    Complete CI/CD pipeline manager for production trading system.
    Handles automated testing, security scanning, deployment, and rollback.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize CI/CD pipeline manager.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Repository and Docker
        self.repo = None
        self.docker_client = None
        
        # Pipeline state
        self.active_runs: Dict[str, PipelineRun] = {}
        self.pipeline_history: List[PipelineRun] = []
        
        # Configuration
        self.pipeline_config = self._load_pipeline_config()
        self.environments = ['development', 'staging', 'production']
        
        # Deployment tracking
        self.deployment_history: List[Dict] = []
        self.rollback_points: Dict[str, str] = {}  # environment -> commit_sha
        
        # Quality gates
        self.quality_gates = {
            'minimum_test_coverage': 80.0,
            'maximum_vulnerabilities': 0,
            'maximum_response_time': 1000,
            'minimum_performance_score': 90
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup CI/CD logger."""
        logger = logging.getLogger('cicd')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('logs/cicd.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        default_config = {
            'stages': {
                'lint': PipelineStage(
                    name='Code Quality Check',
                    commands=[
                        'python -m flake8 --max-line-length=100 --ignore=E203,W503 .',
                        'python -m black --check --diff .',
                        'python -m isort --check-only --diff .'
                    ],
                    timeout=300
                ),
                'security_scan': PipelineStage(
                    name='Security Scanning',
                    commands=[
                        'python -m bandit -r . -f json -o security_report.json',
                        'python -m safety check --json --output safety_report.json'
                    ],
                    timeout=600
                ),
                'unit_tests': PipelineStage(
                    name='Unit Tests',
                    commands=[
                        'python -m pytest tests/unit/ -v --cov=. --cov-report=json --cov-report=html'
                    ],
                    timeout=900
                ),
                'integration_tests': PipelineStage(
                    name='Integration Tests',
                    commands=[
                        'python -m pytest tests/integration/ -v'
                    ],
                    timeout=1800,
                    dependencies=['unit_tests']
                ),
                'performance_tests': PipelineStage(
                    name='Performance Tests',
                    commands=[
                        'python -m pytest tests/performance/ -v --benchmark-json=benchmark.json'
                    ],
                    timeout=1200,
                    dependencies=['unit_tests']
                ),
                'build': PipelineStage(
                    name='Build Application',
                    commands=[
                        'docker build -t omnibeing-trading:latest .',
                        'docker tag omnibeing-trading:latest omnibeing-trading:${COMMIT_SHA}'
                    ],
                    timeout=1800,
                    dependencies=['unit_tests', 'security_scan']
                ),
                'deploy_staging': PipelineStage(
                    name='Deploy to Staging',
                    commands=[
                        'python production_deploy.py --env staging'
                    ],
                    timeout=1800,
                    dependencies=['build', 'integration_tests']
                ),
                'smoke_tests': PipelineStage(
                    name='Smoke Tests',
                    commands=[
                        'python -m pytest tests/smoke/ -v --env=staging'
                    ],
                    timeout=600,
                    dependencies=['deploy_staging']
                ),
                'deploy_production': PipelineStage(
                    name='Deploy to Production',
                    commands=[
                        'python production_deploy.py --env production'
                    ],
                    timeout=1800,
                    dependencies=['smoke_tests'],
                    allow_failure=False
                )
            },
            'branches': {
                'main': ['lint', 'security_scan', 'unit_tests', 'integration_tests', 
                        'performance_tests', 'build', 'deploy_staging', 'smoke_tests', 'deploy_production'],
                'develop': ['lint', 'security_scan', 'unit_tests', 'integration_tests', 'build'],
                'feature/*': ['lint', 'security_scan', 'unit_tests']
            }
        }
        
        # Try to load from file
        config_file = 'ci_config.yaml'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Could not load CI config file: {e}")
        
        return default_config
    
    async def initialize(self):
        """Initialize CI/CD system."""
        try:
            # Initialize Git repository
            self.repo = git.Repo('.')
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Create necessary directories
            os.makedirs('tests/unit', exist_ok=True)
            os.makedirs('tests/integration', exist_ok=True)
            os.makedirs('tests/performance', exist_ok=True)
            os.makedirs('tests/smoke', exist_ok=True)
            os.makedirs('reports', exist_ok=True)
            
            self.logger.info("CI/CD system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CI/CD system: {e}")
            raise
    
    async def trigger_pipeline(self, branch: str = None, commit_sha: str = None) -> str:
        """Trigger CI/CD pipeline for a branch/commit."""
        try:
            if not branch:
                branch = self.repo.active_branch.name
            
            if not commit_sha:
                commit_sha = self.repo.head.commit.hexsha[:8]
            
            # Generate run ID
            run_id = f"{branch}_{commit_sha}_{int(time.time())}"
            
            # Determine stages to run
            stages_to_run = self._get_stages_for_branch(branch)
            
            # Create pipeline run
            pipeline_run = PipelineRun(
                run_id=run_id,
                branch=branch,
                commit_sha=commit_sha,
                started_at=datetime.now(),
                status='running',
                stages={stage: {'status': 'pending', 'start_time': None, 'end_time': None, 'logs': []} 
                        for stage in stages_to_run},
                logs=[]
            )
            
            self.active_runs[run_id] = pipeline_run
            
            self.logger.info(f"Pipeline triggered: {run_id} for {branch}@{commit_sha}")
            
            # Start pipeline execution
            asyncio.create_task(self._execute_pipeline(run_id))
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Error triggering pipeline: {e}")
            raise
    
    def _get_stages_for_branch(self, branch: str) -> List[str]:
        """Get stages to run for a specific branch."""
        branch_config = self.pipeline_config.get('branches', {})
        
        # Check exact match first
        if branch in branch_config:
            return branch_config[branch]
        
        # Check pattern matches
        for pattern, stages in branch_config.items():
            if '*' in pattern:
                prefix = pattern.replace('*', '')
                if branch.startswith(prefix):
                    return stages
        
        # Default stages
        return ['lint', 'security_scan', 'unit_tests']
    
    async def _execute_pipeline(self, run_id: str):
        """Execute pipeline stages."""
        try:
            pipeline_run = self.active_runs[run_id]
            stages_config = self.pipeline_config['stages']
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(list(pipeline_run.stages.keys()), stages_config)
            
            # Execute stages
            executed_stages = set()
            
            while len(executed_stages) < len(pipeline_run.stages):
                # Find stages ready to execute
                ready_stages = []
                for stage_name in pipeline_run.stages:
                    if stage_name in executed_stages:
                        continue
                    
                    stage_config = stages_config.get(stage_name)
                    if not stage_config or not stage_config.dependencies:
                        ready_stages.append(stage_name)
                    else:
                        # Check if all dependencies are completed
                        dependencies_met = all(
                            dep in executed_stages for dep in stage_config.dependencies
                        )
                        if dependencies_met:
                            ready_stages.append(stage_name)
                
                if not ready_stages:
                    # Check for failures in dependencies
                    failed_stages = [
                        stage for stage, info in pipeline_run.stages.items()
                        if info['status'] == 'failed'
                    ]
                    if failed_stages:
                        pipeline_run.status = 'failed'
                        break
                    
                    # Wait for running stages
                    await asyncio.sleep(5)
                    continue
                
                # Execute ready stages in parallel
                tasks = []
                for stage_name in ready_stages:
                    task = asyncio.create_task(self._execute_stage(run_id, stage_name))
                    tasks.append(task)
                
                # Wait for stages to complete
                if tasks:
                    await asyncio.gather(*tasks)
                
                # Mark stages as executed
                for stage_name in ready_stages:
                    if pipeline_run.stages[stage_name]['status'] in ['success', 'failed']:
                        executed_stages.add(stage_name)
            
            # Determine final status
            all_success = all(
                info['status'] == 'success' for info in pipeline_run.stages.values()
            )
            pipeline_run.status = 'success' if all_success else 'failed'
            pipeline_run.completed_at = datetime.now()
            
            # Move to history
            self.pipeline_history.append(pipeline_run)
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            
            self.logger.info(f"Pipeline {run_id} completed with status: {pipeline_run.status}")
            
            # Send notifications
            await self._send_pipeline_notification(pipeline_run)
            
        except Exception as e:
            self.logger.error(f"Error executing pipeline {run_id}: {e}")
            if run_id in self.active_runs:
                self.active_runs[run_id].status = 'failed'
    
    def _build_dependency_graph(self, stages: List[str], stages_config: Dict) -> Dict[str, List[str]]:
        """Build dependency graph for stages."""
        graph = {}
        for stage in stages:
            stage_config = stages_config.get(stage)
            if stage_config and stage_config.dependencies:
                graph[stage] = [dep for dep in stage_config.dependencies if dep in stages]
            else:
                graph[stage] = []
        return graph
    
    async def _execute_stage(self, run_id: str, stage_name: str):
        """Execute a single pipeline stage."""
        try:
            pipeline_run = self.active_runs[run_id]
            stage_info = pipeline_run.stages[stage_name]
            stage_config = self.pipeline_config['stages'][stage_name]
            
            stage_info['status'] = 'running'
            stage_info['start_time'] = datetime.now()
            
            self.logger.info(f"Executing stage {stage_name} for pipeline {run_id}")
            
            # Set environment variables
            env = os.environ.copy()
            env.update({
                'COMMIT_SHA': pipeline_run.commit_sha,
                'BRANCH': pipeline_run.branch,
                'RUN_ID': run_id
            })
            
            # Execute commands
            for command in stage_config.commands:
                # Replace variables in command
                command = command.replace('${COMMIT_SHA}', pipeline_run.commit_sha)
                command = command.replace('${BRANCH}', pipeline_run.branch)
                
                start_time = time.time()
                
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=stage_config.timeout,
                        env=env
                    )
                    
                    duration = time.time() - start_time
                    
                    # Log command output
                    log_entry = {
                        'command': command,
                        'duration': duration,
                        'exit_code': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                    
                    stage_info['logs'].append(log_entry)
                    
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
                    
                except subprocess.TimeoutExpired:
                    stage_info['status'] = 'failed'
                    stage_info['logs'].append({
                        'command': command,
                        'error': f'Command timed out after {stage_config.timeout} seconds'
                    })
                    self.logger.error(f"Stage {stage_name} timed out")
                    return
                
                except subprocess.CalledProcessError as e:
                    if not stage_config.allow_failure:
                        stage_info['status'] = 'failed'
                        self.logger.error(f"Stage {stage_name} failed: {e}")
                        return
                    else:
                        self.logger.warning(f"Stage {stage_name} failed but continuing: {e}")
            
            # Stage completed successfully
            stage_info['status'] = 'success'
            stage_info['end_time'] = datetime.now()
            
            # Process stage results
            await self._process_stage_results(run_id, stage_name)
            
            self.logger.info(f"Stage {stage_name} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in stage {stage_name}: {e}")
            pipeline_run.stages[stage_name]['status'] = 'failed'
    
    async def _process_stage_results(self, run_id: str, stage_name: str):
        """Process results of completed stage."""
        try:
            if stage_name == 'unit_tests':
                await self._process_test_results(run_id)
            elif stage_name == 'security_scan':
                await self._process_security_results(run_id)
            elif stage_name == 'performance_tests':
                await self._process_performance_results(run_id)
            elif stage_name.startswith('deploy_'):
                await self._process_deployment_results(run_id, stage_name)
                
        except Exception as e:
            self.logger.error(f"Error processing results for {stage_name}: {e}")
    
    async def _process_test_results(self, run_id: str):
        """Process unit test results."""
        try:
            # Parse coverage report
            coverage_file = 'coverage.json'
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                    
                    # Check quality gate
                    if total_coverage < self.quality_gates['minimum_test_coverage']:
                        self.logger.warning(f"Test coverage {total_coverage}% below threshold {self.quality_gates['minimum_test_coverage']}%")
                        
                        # Fail pipeline if coverage too low
                        pipeline_run = self.active_runs[run_id]
                        pipeline_run.stages['unit_tests']['status'] = 'failed'
                        pipeline_run.stages['unit_tests']['logs'].append({
                            'error': f'Test coverage {total_coverage}% below required {self.quality_gates["minimum_test_coverage"]}%'
                        })
                    
        except Exception as e:
            self.logger.error(f"Error processing test results: {e}")
    
    async def _process_security_results(self, run_id: str):
        """Process security scan results."""
        try:
            # Parse security reports
            vulnerabilities = 0
            
            # Parse Bandit report
            bandit_file = 'security_report.json'
            if os.path.exists(bandit_file):
                with open(bandit_file, 'r') as f:
                    bandit_data = json.load(f)
                    vulnerabilities += len(bandit_data.get('results', []))
            
            # Parse Safety report
            safety_file = 'safety_report.json'
            if os.path.exists(safety_file):
                with open(safety_file, 'r') as f:
                    safety_data = json.load(f)
                    vulnerabilities += len(safety_data)
            
            # Check quality gate
            if vulnerabilities > self.quality_gates['maximum_vulnerabilities']:
                self.logger.error(f"Security scan found {vulnerabilities} vulnerabilities")
                
                # Fail pipeline if vulnerabilities found
                pipeline_run = self.active_runs[run_id]
                pipeline_run.stages['security_scan']['status'] = 'failed'
                pipeline_run.stages['security_scan']['logs'].append({
                    'error': f'Found {vulnerabilities} security vulnerabilities'
                })
                
        except Exception as e:
            self.logger.error(f"Error processing security results: {e}")
    
    async def _process_performance_results(self, run_id: str):
        """Process performance test results."""
        try:
            # Parse benchmark results
            benchmark_file = 'benchmark.json'
            if os.path.exists(benchmark_file):
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                    
                    # Check performance thresholds
                    for benchmark in benchmark_data.get('benchmarks', []):
                        if benchmark.get('stats', {}).get('mean', 0) > self.quality_gates['maximum_response_time'] / 1000:
                            self.logger.warning(f"Performance test {benchmark['name']} exceeded threshold")
                            
        except Exception as e:
            self.logger.error(f"Error processing performance results: {e}")
    
    async def _process_deployment_results(self, run_id: str, stage_name: str):
        """Process deployment results."""
        try:
            environment = stage_name.replace('deploy_', '')
            pipeline_run = self.active_runs[run_id]
            
            # Record successful deployment
            deployment_record = {
                'environment': environment,
                'commit_sha': pipeline_run.commit_sha,
                'branch': pipeline_run.branch,
                'deployed_at': datetime.now(),
                'run_id': run_id
            }
            
            self.deployment_history.append(deployment_record)
            
            # Update rollback point
            self.rollback_points[environment] = pipeline_run.commit_sha
            
            self.logger.info(f"Deployment to {environment} recorded: {pipeline_run.commit_sha}")
            
        except Exception as e:
            self.logger.error(f"Error processing deployment results: {e}")
    
    async def _send_pipeline_notification(self, pipeline_run: PipelineRun):
        """Send pipeline completion notification."""
        try:
            status_emoji = "✅" if pipeline_run.status == 'success' else "❌"
            duration = (pipeline_run.completed_at - pipeline_run.started_at).total_seconds()
            
            message = f"{status_emoji} Pipeline {pipeline_run.run_id} {pipeline_run.status}\n"
            message += f"Branch: {pipeline_run.branch}\n"
            message += f"Commit: {pipeline_run.commit_sha}\n"
            message += f"Duration: {duration:.1f}s\n"
            
            # Add failed stages info
            failed_stages = [
                stage for stage, info in pipeline_run.stages.items()
                if info['status'] == 'failed'
            ]
            if failed_stages:
                message += f"Failed stages: {', '.join(failed_stages)}\n"
            
            self.logger.info(f"Pipeline notification: {message}")
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
    
    async def rollback_deployment(self, environment: str, target_commit: str = None) -> bool:
        """Rollback deployment to previous version."""
        try:
            if not target_commit:
                # Find previous successful deployment
                env_deployments = [
                    d for d in self.deployment_history
                    if d['environment'] == environment
                ]
                
                if len(env_deployments) < 2:
                    raise ValueError(f"No previous deployment found for {environment}")
                
                # Get second-to-last deployment
                target_commit = env_deployments[-2]['commit_sha']
            
            self.logger.info(f"Rolling back {environment} to {target_commit}")
            
            # Create rollback pipeline
            run_id = await self.trigger_rollback_pipeline(environment, target_commit)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rolling back {environment}: {e}")
            return False
    
    async def trigger_rollback_pipeline(self, environment: str, target_commit: str) -> str:
        """Trigger rollback pipeline."""
        try:
            # Generate rollback run ID
            run_id = f"rollback_{environment}_{target_commit}_{int(time.time())}"
            
            # Create simplified pipeline for rollback
            pipeline_run = PipelineRun(
                run_id=run_id,
                branch='rollback',
                commit_sha=target_commit,
                started_at=datetime.now(),
                status='running',
                stages={
                    'rollback': {
                        'status': 'pending',
                        'start_time': None,
                        'end_time': None,
                        'logs': []
                    }
                },
                logs=[]
            )
            
            self.active_runs[run_id] = pipeline_run
            
            # Execute rollback
            asyncio.create_task(self._execute_rollback(run_id, environment, target_commit))
            
            return run_id
            
        except Exception as e:
            self.logger.error(f"Error triggering rollback pipeline: {e}")
            raise
    
    async def _execute_rollback(self, run_id: str, environment: str, target_commit: str):
        """Execute rollback process."""
        try:
            pipeline_run = self.active_runs[run_id]
            stage_info = pipeline_run.stages['rollback']
            
            stage_info['status'] = 'running'
            stage_info['start_time'] = datetime.now()
            
            # Checkout target commit
            self.repo.git.checkout(target_commit)
            
            # Deploy previous version
            command = f"python production_deploy.py --env {environment}"
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode == 0:
                stage_info['status'] = 'success'
                pipeline_run.status = 'success'
                
                # Update rollback point
                self.rollback_points[environment] = target_commit
                
                self.logger.info(f"Rollback to {target_commit} completed successfully")
                
            else:
                stage_info['status'] = 'failed'
                pipeline_run.status = 'failed'
                self.logger.error(f"Rollback failed: {result.stderr}")
            
            stage_info['end_time'] = datetime.now()
            pipeline_run.completed_at = datetime.now()
            
            # Move to history
            self.pipeline_history.append(pipeline_run)
            del self.active_runs[run_id]
            
        except Exception as e:
            self.logger.error(f"Error executing rollback: {e}")
            pipeline_run.status = 'failed'
    
    async def get_pipeline_status(self, run_id: str) -> Optional[Dict]:
        """Get status of a specific pipeline run."""
        if run_id in self.active_runs:
            return asdict(self.active_runs[run_id])
        
        # Check history
        for run in self.pipeline_history:
            if run.run_id == run_id:
                return asdict(run)
        
        return None
    
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status for all environments."""
        status = {}
        
        for env in self.environments:
            env_deployments = [
                d for d in self.deployment_history
                if d['environment'] == env
            ]
            
            if env_deployments:
                latest = env_deployments[-1]
                status[env] = {
                    'current_commit': latest['commit_sha'],
                    'deployed_at': latest['deployed_at'].isoformat(),
                    'branch': latest['branch'],
                    'rollback_available': len(env_deployments) > 1
                }
            else:
                status[env] = {
                    'current_commit': None,
                    'deployed_at': None,
                    'branch': None,
                    'rollback_available': False
                }
        
        return status
    
    async def create_release(self, version: str, description: str = "") -> bool:
        """Create a release tag and trigger production deployment."""
        try:
            # Create Git tag
            tag = self.repo.create_tag(version, message=description)
            
            # Push tag to remote
            origin = self.repo.remote('origin')
            origin.push(tag)
            
            # Trigger production deployment
            run_id = await self.trigger_pipeline('main', tag.commit.hexsha[:8])
            
            self.logger.info(f"Release {version} created and deployment triggered: {run_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating release {version}: {e}")
            return False
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get code quality metrics."""
        # This would typically integrate with SonarQube or similar
        return {
            'test_coverage': 85.2,
            'code_smells': 12,
            'security_hotspots': 2,
            'technical_debt': '2h 30m',
            'maintainability_rating': 'A',
            'reliability_rating': 'A',
            'security_rating': 'A'
        }
    
    async def cleanup_old_data(self):
        """Clean up old pipeline data."""
        try:
            # Keep only recent pipeline history (last 100 runs)
            if len(self.pipeline_history) > 100:
                self.pipeline_history = self.pipeline_history[-100:]
            
            # Keep only recent deployment history (last 50 deployments)
            if len(self.deployment_history) > 50:
                self.deployment_history = self.deployment_history[-50:]
            
            # Clean up old report files
            reports_dir = Path('reports')
            if reports_dir.exists():
                cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
                for file_path in reports_dir.glob('*'):
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
            
            self.logger.info("CI/CD data cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main CI/CD entry point."""
    from production_config import get_production_config
    
    config = get_production_config()
    ci_cd = ContinuousIntegration(config)
    
    await ci_cd.initialize()
    
    # Example: trigger pipeline for current branch
    run_id = await ci_cd.trigger_pipeline()
    print(f"Pipeline triggered: {run_id}")
    
    # Wait for completion (in real implementation, this would be event-driven)
    while run_id in ci_cd.active_runs:
        await asyncio.sleep(10)
        status = await ci_cd.get_pipeline_status(run_id)
        print(f"Pipeline status: {status['status']}")


if __name__ == "__main__":
    asyncio.run(main())