#!/usr/bin/env python3
"""
Production Deployment Suite Integration Test.
Validates all components work together correctly.
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_production_config():
    """Test production configuration loading."""
    try:
        from production_config import ProductionConfig, Environment
        
        print("🔧 Testing Production Config...")
        
        # Test basic config loading
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        
        # Test configuration validation
        assert config.environment == Environment.DEVELOPMENT
        assert hasattr(config, 'database')
        assert hasattr(config, 'redis')
        assert hasattr(config, 'security')
        assert hasattr(config, 'monitoring')
        assert hasattr(config, 'scaling')
        
        print("✅ Production Config: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Production Config: FAILED - {e}")
        return False

async def test_monitoring_suite():
    """Test monitoring suite initialization."""
    try:
        from production_config import ProductionConfig, Environment
        from monitoring_suite import MonitoringSuite
        
        print("📊 Testing Monitoring Suite...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        monitoring = MonitoringSuite(config)
        
        # Test metrics collection (without Redis/DB)
        assert hasattr(monitoring, 'logger')
        assert hasattr(monitoring, 'config')
        assert monitoring.collection_interval == 30
        
        # Test alert thresholds
        assert len(monitoring.alert_thresholds) > 0
        
        print("✅ Monitoring Suite: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring Suite: FAILED - {e}")
        return False

async def test_security_hardening():
    """Test security hardening components."""
    try:
        from production_config import ProductionConfig, Environment
        from security_hardening import SecurityManager
        
        print("🔒 Testing Security Hardening...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        security = SecurityManager(config)
        
        # Test password hashing
        password = "test_password_123"
        hashed = security.hash_password(password)
        assert security.verify_password(password, hashed)
        
        # Test token generation
        token = security.generate_secure_token()
        assert len(token) > 0
        
        # Test JWT token creation
        jwt_token = security.create_jwt_token("test_user", ["trading"])
        assert len(jwt_token) > 0
        
        print("✅ Security Hardening: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Security Hardening: FAILED - {e}")
        return False

async def test_auto_scaling():
    """Test auto-scaling manager."""
    try:
        from production_config import ProductionConfig, Environment
        from auto_scaling import AutoScalingManager
        
        print("📈 Testing Auto-Scaling...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        autoscaler = AutoScalingManager(config)
        
        # Test initialization
        assert hasattr(autoscaler, 'config')
        assert hasattr(autoscaler, 'managed_services')
        assert len(autoscaler.managed_services) > 0
        
        # Test circuit breakers
        assert hasattr(autoscaler, 'circuit_breakers')
        
        print("✅ Auto-Scaling: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Auto-Scaling: FAILED - {e}")
        return False

async def test_continuous_integration():
    """Test CI/CD pipeline."""
    try:
        from production_config import ProductionConfig, Environment
        from continuous_integration import ContinuousIntegration
        
        print("🔄 Testing CI/CD Pipeline...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        ci_cd = ContinuousIntegration(config)
        
        # Test pipeline configuration
        assert hasattr(ci_cd, 'pipeline_config')
        assert 'stages' in ci_cd.pipeline_config
        assert len(ci_cd.environments) > 0
        
        # Test quality gates
        assert hasattr(ci_cd, 'quality_gates')
        assert 'minimum_test_coverage' in ci_cd.quality_gates
        
        print("✅ CI/CD Pipeline: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ CI/CD Pipeline: FAILED - {e}")
        return False

async def test_live_trading_manager():
    """Test live trading manager."""
    try:
        from production_config import ProductionConfig, Environment
        from live_trading_manager import LiveTradingManager, OrderType, OrderSide
        
        print("💰 Testing Live Trading Manager...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        trading = LiveTradingManager(config)
        
        # Test enums
        assert OrderType.MARKET.value == "market"
        assert OrderSide.BUY.value == "buy"
        
        # Test trading manager attributes
        assert hasattr(trading, 'config')
        assert hasattr(trading, 'exchange_configs')
        assert hasattr(trading, 'risk_limits')
        assert hasattr(trading, 'performance_metrics')
        
        print("✅ Live Trading Manager: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Live Trading Manager: FAILED - {e}")
        return False

async def test_enterprise_api():
    """Test enterprise API."""
    try:
        from production_config import ProductionConfig, Environment
        from enterprise_api import EnterpriseAPI
        
        print("🌐 Testing Enterprise API...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        api = EnterpriseAPI(config)
        
        # Test FastAPI app
        assert hasattr(api, 'app')
        assert api.app.title == "OmniBeing Trading System API"
        
        # Test connection manager
        assert hasattr(api, 'connection_manager')
        
        # Test API stats
        assert hasattr(api, 'api_stats')
        assert 'requests_total' in api.api_stats
        
        print("✅ Enterprise API: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enterprise API: FAILED - {e}")
        return False

async def test_compliance_reporting():
    """Test compliance reporting system."""
    try:
        from production_config import ProductionConfig, Environment
        from compliance_reporting import ComplianceReporting, ReportType, ComplianceLevel
        
        print("📋 Testing Compliance Reporting...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        compliance = ComplianceReporting(config)
        
        # Test enums
        assert ReportType.TRADE_REPORT.value == "trade_report"
        assert ComplianceLevel.CRITICAL.value == "critical"
        
        # Test compliance configuration
        assert hasattr(compliance, 'reporting_thresholds')
        assert hasattr(compliance, 'tax_rates')
        assert hasattr(compliance, 'aml_thresholds')
        
        # Test report templates
        assert hasattr(compliance, 'report_templates')
        assert 'trade_report' in compliance.report_templates
        
        print("✅ Compliance Reporting: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Compliance Reporting: FAILED - {e}")
        return False

async def test_deployment_orchestrator():
    """Test deployment orchestrator."""
    try:
        from production_config import ProductionConfig, Environment
        from production_deploy import ProductionDeployer
        
        print("🚀 Testing Deployment Orchestrator...")
        
        config = ProductionConfig(environment=Environment.DEVELOPMENT)
        deployer = ProductionDeployer(environment=Environment.DEVELOPMENT)
        
        # Test deployment phases
        assert hasattr(deployer, 'phases')
        assert len(deployer.phases) > 0
        
        # Check phase structure
        for phase_name, phase_func in deployer.phases:
            assert isinstance(phase_name, str)
            assert callable(phase_func)
        
        print("✅ Deployment Orchestrator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Deployment Orchestrator: FAILED - {e}")
        return False

async def test_docker_compose_validation():
    """Test Docker Compose configuration."""
    try:
        import yaml
        
        print("🐳 Testing Docker Compose...")
        
        # Load and validate docker-compose.yml
        with open('docker-compose.yml', 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Validate structure
        assert 'version' in compose_config
        assert 'services' in compose_config
        assert 'volumes' in compose_config
        assert 'networks' in compose_config
        
        # Check required services
        required_services = [
            'trading-system', 'postgresql', 'redis', 'nginx',
            'prometheus', 'grafana', 'elasticsearch'
        ]
        
        for service in required_services:
            assert service in compose_config['services'], f"Missing service: {service}"
        
        print("✅ Docker Compose: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Docker Compose: FAILED - {e}")
        return False

async def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("🧪 OMNIBEING PRODUCTION DEPLOYMENT SUITE TESTS")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        test_production_config,
        test_monitoring_suite,
        test_security_hardening,
        test_auto_scaling,
        test_continuous_integration,
        test_live_trading_manager,
        test_enterprise_api,
        test_compliance_reporting,
        test_deployment_orchestrator,
        test_docker_compose_validation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAILED - {e}")
            failed += 1
        
        print()
    
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print(f"⏱️  Duration: {time.time() - start_time:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Production suite is ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix issues before deployment.")
        return False

def main():
    """Main test entry point."""
    global start_time
    start_time = time.time()
    
    try:
        success = asyncio.run(run_integration_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test runner error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()