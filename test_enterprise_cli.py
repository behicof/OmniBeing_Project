"""
Tests for the BehicoF CLI Enterprise Deployment System.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch

from behicof_cli import BehicofCLI


class TestBehicofCLI:
    """Test enterprise CLI functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.cli = BehicofCLI()
    
    def test_cli_initialization(self):
        """Test CLI initializes with enterprise modules."""
        assert len(self.cli.enterprise_modules) == 6
        assert 'risk_manager' in self.cli.enterprise_modules
        assert 'compliance' in self.cli.enterprise_modules
        assert 'live_trading' in self.cli.enterprise_modules
        assert 'arbitrage' in self.cli.enterprise_modules
        assert 'ai_strategies' in self.cli.enterprise_modules
        assert 'reporting' in self.cli.enterprise_modules
        
        # All modules should start disabled
        for module in self.cli.enterprise_modules.values():
            assert module['status'] == 'disabled'
    
    def test_sla_thresholds(self):
        """Test SLA thresholds match enterprise requirements."""
        thresholds = self.cli.sla_thresholds
        assert thresholds['processing_delay_ms'] == 5
        assert thresholds['trading_error_percentage'] == 0.1
        assert thresholds['ram_usage_percentage'] == 85
    
    def test_module_priorities(self):
        """Test module priorities for phased activation."""
        # Phase 1 modules (priority 1)
        assert self.cli.enterprise_modules['risk_manager']['priority'] == 1
        assert self.cli.enterprise_modules['compliance']['priority'] == 1
        
        # Phase 2 modules (priority 2)
        assert self.cli.enterprise_modules['live_trading']['priority'] == 2
        assert self.cli.enterprise_modules['arbitrage']['priority'] == 2
        
        # Phase 3 modules (priority 3)
        assert self.cli.enterprise_modules['ai_strategies']['priority'] == 3
        assert self.cli.enterprise_modules['reporting']['priority'] == 3
    
    def test_enable_single_module(self):
        """Test enabling a single enterprise module."""
        result = self.cli.enable_modules(['risk_manager'])
        assert result is True
        assert self.cli.enterprise_modules['risk_manager']['status'] == 'enabled'
    
    def test_enable_phase_modules(self):
        """Test enabling modules by phase."""
        # Enable Phase 1 modules
        result = self.cli.enable_modules(['risk_manager', 'compliance'])
        assert result is True
        assert self.cli.enterprise_modules['risk_manager']['status'] == 'enabled'
        assert self.cli.enterprise_modules['compliance']['status'] == 'enabled'
    
    def test_enable_invalid_module(self):
        """Test handling of invalid module names."""
        result = self.cli.enable_modules(['invalid_module'])
        assert result is False
    
    def test_enable_mixed_phases(self):
        """Test enabling modules from different phases in correct order."""
        result = self.cli.enable_modules(['ai_strategies', 'risk_manager', 'live_trading'])
        assert result is True
        
        # All should be enabled regardless of input order
        assert self.cli.enterprise_modules['risk_manager']['status'] == 'enabled'
        assert self.cli.enterprise_modules['live_trading']['status'] == 'enabled'
        assert self.cli.enterprise_modules['ai_strategies']['status'] == 'enabled'
    
    def test_status_output(self):
        """Test status output format and content."""
        status = self.cli.status()
        
        assert 'timestamp' in status
        assert status['platform'] == 'OmniBeing Enterprise'
        assert status['version'] == '18.0 Enterprise'
        assert 'modules' in status
        assert 'sla_thresholds' in status
        assert 'enterprise_settings' in status
        
        # Check all modules present
        assert len(status['modules']) == 6
        assert len(status['enterprise_settings']) == 6
    
    def test_stress_test_execution(self):
        """Test stress testing functionality."""
        result = self.cli.stress_test(multiplier=5)
        assert result is True
        
        result = self.cli.stress_test(multiplier=20)
        assert result is True
    
    def test_configure_monitoring(self):
        """Test monitoring configuration."""
        result = self.cli.configure_monitoring()
        assert result is True
        
        # Check configuration was set
        assert self.cli.config.get('enterprise.monitoring.enabled') is True
        assert self.cli.config.get('enterprise.monitoring.real_time') is True
        assert self.cli.config.get('enterprise.monitoring.processing_delay_ms') == 5
        assert self.cli.config.get('enterprise.monitoring.trading_error_percentage') == 0.1
        assert self.cli.config.get('enterprise.monitoring.ram_usage_percentage') == 85
    
    @patch('behicof_cli.time.sleep')  # Speed up tests by mocking sleep
    def test_full_deployment_sequence(self, mock_sleep):
        """Test complete enterprise deployment sequence."""
        # Phase 1
        result1 = self.cli.enable_modules(['risk_manager', 'compliance'])
        assert result1 is True
        
        # Phase 2
        result2 = self.cli.enable_modules(['live_trading', 'arbitrage'])
        assert result2 is True
        
        # Phase 3
        result3 = self.cli.enable_modules(['ai_strategies', 'reporting'])
        assert result3 is True
        
        # Configure monitoring
        result4 = self.cli.configure_monitoring()
        assert result4 is True
        
        # Run stress test
        result5 = self.cli.stress_test()
        assert result5 is True
        
        # Check final status
        status = self.cli.status()
        for module in status['enterprise_settings']:
            assert status['enterprise_settings'][module] is True


class TestCLIModuleActivation:
    """Test individual module activation functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.cli = BehicofCLI()
    
    def test_activate_risk_manager(self):
        """Test risk manager activation."""
        result = self.cli._activate_risk_manager()
        assert result is True
        assert self.cli.config.get('enterprise.risk_manager.enabled') is True
        assert self.cli.config.get('enterprise.risk_manager.sla_monitoring') is True
    
    def test_activate_compliance(self):
        """Test compliance module activation."""
        result = self.cli._activate_compliance()
        assert result is True
        assert self.cli.config.get('enterprise.compliance.enabled') is True
        assert self.cli.config.get('enterprise.compliance.iso27001_mode') is True
    
    def test_activate_live_trading(self):
        """Test live trading activation."""
        result = self.cli._activate_live_trading()
        assert result is True
        assert self.cli.config.get('enterprise.live_trading.enabled') is True
        assert self.cli.config.get('enterprise.live_trading.institutional_mode') is True
    
    def test_activate_arbitrage(self):
        """Test arbitrage module activation."""
        result = self.cli._activate_arbitrage()
        assert result is True
        assert self.cli.config.get('enterprise.arbitrage.enabled') is True
        assert self.cli.config.get('enterprise.arbitrage.multi_exchange') is True
    
    def test_activate_ai_strategies(self):
        """Test AI strategies activation."""
        result = self.cli._activate_ai_strategies()
        assert result is True
        assert self.cli.config.get('enterprise.ai_strategies.enabled') is True
        assert self.cli.config.get('enterprise.ai_strategies.deep_learning') is True
    
    def test_activate_reporting(self):
        """Test reporting module activation."""
        result = self.cli._activate_reporting()
        assert result is True
        assert self.cli.config.get('enterprise.reporting.enabled') is True
        assert self.cli.config.get('enterprise.reporting.institutional_format') is True


class TestCLICommandValidation:
    """Test CLI command validation and error handling."""
    
    def setup_method(self):
        """Setup test environment."""
        self.cli = BehicofCLI()
    
    def test_empty_module_list(self):
        """Test handling of empty module list."""
        result = self.cli.enable_modules([])
        assert result is True  # Should succeed with no action
    
    def test_duplicate_modules(self):
        """Test handling of duplicate modules in list."""
        result = self.cli.enable_modules(['risk_manager', 'risk_manager'])
        assert result is True
        assert self.cli.enterprise_modules['risk_manager']['status'] == 'enabled'
    
    def test_module_validation(self):
        """Test module name validation."""
        valid_modules = ['risk_manager', 'compliance', 'live_trading', 
                        'arbitrage', 'ai_strategies', 'reporting']
        
        for module in valid_modules:
            result = self.cli.enable_modules([module])
            assert result is True, f"Failed to enable valid module: {module}"
            
            # Reset for next test
            self.cli.enterprise_modules[module]['status'] = 'disabled'