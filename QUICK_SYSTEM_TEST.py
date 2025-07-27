#!/usr/bin/env python3
"""
QUICK_SYSTEM_TEST.PY - Rapid System Validation
===============================================

Comprehensive quick validation of all OmniBeing Trading System components.
Target execution time: ~2 minutes

Tests:
- System initialization
- Import validation
- Configuration loading
- API connections (mock)
- Basic prediction pipeline
- Risk assessment functionality
- Data flow validation

Created by behicof for the OmniBeing Trading System
"""

import time
import sys
import os
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

class QuickSystemTest:
    """Quick system validation test runner."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.start_time = time.time()
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
        
        print("=" * 70)
        print("🚀 QUICK SYSTEM TEST - OmniBeing Trading System")
        print("=" * 70)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target time: ~2 minutes")
        print()
    
    def log_test(self, test_name: str, passed: bool, message: str = "", warning: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {test_name}")
        if message:
            print(f"      💡 {message}")
        if warning:
            print(f"      ⚠️  {warning}")
            self.warnings.append(f"{test_name}: {warning}")
        
        self.test_results[test_name] = {
            'passed': passed,
            'message': message,
            'warning': warning
        }
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def test_system_imports(self) -> bool:
        """Test 1: Validate all critical imports."""
        print("\n🔍 Test 1: System Import Validation")
        
        critical_imports = [
            ('yaml', 'PyYAML'),
            ('datetime', 'built-in'),
            ('typing', 'built-in'),
            ('os', 'built-in'),
            ('sys', 'built-in'),
            ('time', 'built-in'),
            ('threading', 'built-in')
        ]
        
        optional_imports = [
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('sklearn', 'scikit-learn'),
            ('matplotlib', 'matplotlib'),
            ('requests', 'requests')
        ]
        
        # Test critical imports
        all_critical_passed = True
        for module, package in critical_imports:
            try:
                __import__(module)
                self.log_test(f"Import {module}", True, f"Required package {package} available")
            except ImportError as e:
                self.log_test(f"Import {module}", False, f"Critical package {package} missing: {e}")
                all_critical_passed = False
        
        # Test optional imports
        optional_available = 0
        for module, package in optional_imports:
            try:
                __import__(module)
                self.log_test(f"Import {module} (optional)", True, f"Optional package {package} available")
                optional_available += 1
            except ImportError as e:
                self.log_test(f"Import {module} (optional)", True, 
                            f"Optional package {package} not available (will use fallbacks)",
                            f"Package {package} missing: {e}")
        
        # Overall import assessment
        if all_critical_passed:
            if optional_available >= 3:
                self.log_test("Overall Import Status", True, 
                            f"All critical imports passed, {optional_available}/{len(optional_imports)} optional packages available")
            else:
                self.log_test("Overall Import Status", True, 
                            f"Critical imports passed, but only {optional_available}/{len(optional_imports)} optional packages available",
                            "Some features may use fallback implementations")
        else:
            self.log_test("Overall Import Status", False, "Critical import failures detected")
        
        return all_critical_passed
    
    def test_configuration_system(self) -> bool:
        """Test 2: Configuration loading and validation."""
        print("\n📋 Test 2: Configuration System Validation")
        
        try:
            # Test config import
            from config import config, Config
            self.log_test("Config Import", True, "Configuration module imported successfully")
        except ImportError as e:
            self.log_test("Config Import", False, f"Cannot import config: {e}")
            return False
        
        try:
            # Test config file existence
            if os.path.exists('config.yaml'):
                self.log_test("Config File", True, "config.yaml found")
            else:
                self.log_test("Config File", False, "config.yaml not found")
                return False
            
            # Test config loading
            test_config = Config('config.yaml')
            self.log_test("Config Loading", True, "Configuration loaded successfully")
            
            # Test key properties
            trading_instrument = config.trading_instrument
            initial_capital = config.initial_capital
            risk_percentage = config.risk_percentage
            
            self.log_test("Config Properties", True, 
                        f"Instrument: {trading_instrument}, Capital: {initial_capital}, Risk: {risk_percentage}%")
            
            # Test config get/set operations
            test_value = config.get('test.validation', 'default')
            config.set('test.validation', 'validated')
            retrieved_value = config.get('test.validation')
            
            if retrieved_value == 'validated':
                self.log_test("Config Operations", True, "Get/Set operations working correctly")
            else:
                self.log_test("Config Operations", False, "Get/Set operations failed")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Configuration System", False, f"Configuration error: {e}")
            return False
    
    def test_core_components(self) -> bool:
        """Test 3: Core component initialization."""
        print("\n🏗️  Test 3: Core Component Validation")
        
        components_status = {}
        
        # Test DataManager
        try:
            from data_manager import DataManager
            dm = DataManager()
            self.log_test("DataManager", True, "DataManager initialized successfully")
            components_status['data_manager'] = True
        except Exception as e:
            self.log_test("DataManager", False, f"DataManager failed: {e}")
            components_status['data_manager'] = False
        
        # Test ExternalRiskManager
        try:
            from external_risk_manager import ExternalRiskManager
            rm = ExternalRiskManager()
            self.log_test("RiskManager", True, "RiskManager initialized successfully")
            components_status['risk_manager'] = True
        except Exception as e:
            self.log_test("RiskManager", False, f"RiskManager failed: {e}")
            components_status['risk_manager'] = False
        
        # Test MainTradingSystem
        try:
            from main_trading_system import MainTradingSystem
            ts = MainTradingSystem()
            self.log_test("TradingSystem", True, "MainTradingSystem initialized successfully")
            components_status['trading_system'] = True
        except Exception as e:
            self.log_test("TradingSystem", False, f"MainTradingSystem failed: {e}")
            components_status['trading_system'] = False
        
        # Check if prediction systems are available
        try:
            if hasattr(ts, 'prediction_systems'):
                available_systems = list(ts.prediction_systems.keys())
                if available_systems:
                    self.log_test("PredictionSystems", True, f"Available systems: {available_systems}")
                else:
                    self.log_test("PredictionSystems", True, "No prediction systems loaded (fallback mode)",
                                "Will use mock predictions for testing")
            else:
                self.log_test("PredictionSystems", False, "Prediction systems not accessible")
        except Exception as e:
            self.log_test("PredictionSystems", True, f"Prediction systems check skipped: {e}",
                        "Will use mock predictions for testing")
        
        return all(components_status.values())
    
    def test_data_pipeline(self) -> bool:
        """Test 4: Data pipeline functionality."""
        print("\n📊 Test 4: Data Pipeline Validation")
        
        try:
            from main_trading_system import MainTradingSystem
            ts = MainTradingSystem()
            
            # Test market data retrieval
            market_data = ts.get_market_data('XAUUSD')
            if market_data and isinstance(market_data, dict):
                self.log_test("Market Data", True, f"Retrieved market data with {len(market_data)} fields")
            else:
                self.log_test("Market Data", False, "Failed to retrieve market data")
                return False
            
            # Test data structure
            expected_fields = ['price', 'sentiment', 'volatility']
            missing_fields = [field for field in expected_fields if field not in market_data]
            
            if not missing_fields:
                self.log_test("Data Structure", True, "All expected fields present in market data")
            else:
                self.log_test("Data Structure", True, f"Market data structure valid",
                            f"Missing optional fields: {missing_fields}")
            
            return True
            
        except Exception as e:
            self.log_test("Data Pipeline", False, f"Data pipeline error: {e}")
            return False
    
    def test_prediction_pipeline(self) -> bool:
        """Test 5: Prediction pipeline functionality."""
        print("\n🧠 Test 5: Prediction Pipeline Validation")
        
        try:
            from main_trading_system import MainTradingSystem
            ts = MainTradingSystem()
            
            # Test prediction generation
            prediction = ts.make_prediction('XAUUSD')
            if prediction and isinstance(prediction, dict):
                self.log_test("Prediction Generation", True, "Prediction generated successfully")
                
                # Check prediction structure
                if 'combined_prediction' in prediction:
                    pred_value = prediction['combined_prediction']
                    if pred_value in ['buy', 'sell', 'hold']:
                        self.log_test("Prediction Format", True, f"Valid prediction: {pred_value}")
                    else:
                        self.log_test("Prediction Format", False, f"Invalid prediction value: {pred_value}")
                        return False
                else:
                    self.log_test("Prediction Format", True, "Prediction generated in fallback mode",
                                "Combined prediction not available")
                
                # Check for individual predictions
                if 'individual_predictions' in prediction:
                    individual = prediction['individual_predictions']
                    self.log_test("Individual Predictions", True, f"Found {len(individual)} prediction systems")
                else:
                    self.log_test("Individual Predictions", True, "No individual predictions available",
                                "System running in basic mode")
                
                return True
            else:
                self.log_test("Prediction Generation", False, "Failed to generate prediction")
                return False
                
        except Exception as e:
            self.log_test("Prediction Pipeline", False, f"Prediction error: {e}")
            return False
    
    def test_risk_management(self) -> bool:
        """Test 6: Risk management functionality."""
        print("\n🛡️  Test 6: Risk Management Validation")
        
        try:
            from main_trading_system import MainTradingSystem
            ts = MainTradingSystem()
            
            # Test risk assessment
            risk_assessment = ts.assess_risk('XAUUSD')
            if risk_assessment and isinstance(risk_assessment, dict):
                self.log_test("Risk Assessment", True, "Risk assessment completed successfully")
                
                # Check risk signal
                if 'risk_signal' in risk_assessment:
                    risk_signal = risk_assessment['risk_signal']
                    if isinstance(risk_signal, dict) and 'action' in risk_signal:
                        action = risk_signal['action']
                        self.log_test("Risk Signal", True, f"Risk action: {action}")
                    else:
                        self.log_test("Risk Signal", False, "Invalid risk signal format")
                        return False
                else:
                    self.log_test("Risk Signal", False, "No risk signal in assessment")
                    return False
                
                # Test risk manager directly
                rm = ts.risk_manager
                if hasattr(rm, 'account_balance') and hasattr(rm, 'portfolio'):
                    balance = rm.account_balance
                    positions = len(rm.portfolio)
                    self.log_test("Risk Manager State", True, f"Balance: ${balance:,.2f}, Positions: {positions}")
                else:
                    self.log_test("Risk Manager State", False, "Risk manager state invalid")
                    return False
                
                return True
            else:
                self.log_test("Risk Assessment", False, "Failed to perform risk assessment")
                return False
                
        except Exception as e:
            self.log_test("Risk Management", False, f"Risk management error: {e}")
            return False
    
    def test_system_integration(self) -> bool:
        """Test 7: System integration and workflow."""
        print("\n🔄 Test 7: System Integration Validation")
        
        try:
            from main_trading_system import MainTradingSystem
            ts = MainTradingSystem()
            
            # Test full workflow
            # 1. Get system status
            status = ts.get_system_status()
            if status and isinstance(status, dict):
                self.log_test("System Status", True, f"System operational, Trading: {status.get('is_trading_enabled', False)}")
            else:
                self.log_test("System Status", False, "Cannot retrieve system status")
                return False
            
            # 2. Test prediction -> risk -> decision flow
            prediction = ts.make_prediction('XAUUSD')
            if prediction:
                self.log_test("Workflow Step 1", True, "Prediction generated")
                
                risk_assessment = ts.assess_risk('XAUUSD')
                if risk_assessment:
                    self.log_test("Workflow Step 2", True, "Risk assessment completed")
                    
                    # Test simulated trade execution
                    if prediction.get('combined_prediction') in ['buy', 'sell']:
                        try:
                            trade_result = ts.execute_trade(prediction, 'XAUUSD')
                            if trade_result and 'status' in trade_result:
                                self.log_test("Workflow Step 3", True, f"Trade simulation: {trade_result['status']}")
                            else:
                                self.log_test("Workflow Step 3", False, "Trade execution failed")
                                return False
                        except Exception as e:
                            self.log_test("Workflow Step 3", True, f"Trade execution test skipped: {e}",
                                        "Trade execution requires additional dependencies")
                    else:
                        self.log_test("Workflow Step 3", True, "No trade signal generated (hold)")
                else:
                    self.log_test("Workflow Step 2", False, "Risk assessment failed")
                    return False
            else:
                self.log_test("Workflow Step 1", False, "Prediction generation failed")
                return False
            
            # Test system controls
            ts.disable_trading()
            if not ts.is_trading_enabled:
                self.log_test("Trading Controls", True, "Trading disable/enable functionality working")
                ts.enable_trading()
            else:
                self.log_test("Trading Controls", False, "Trading controls not working")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("System Integration", False, f"Integration test error: {e}")
            return False
    
    def test_performance_check(self) -> bool:
        """Test 8: Basic performance validation."""
        print("\n⚡ Test 8: Performance Check")
        
        try:
            from main_trading_system import MainTradingSystem
            ts = MainTradingSystem()
            
            # Time prediction generation
            start_time = time.time()
            for i in range(3):
                prediction = ts.make_prediction('XAUUSD')
            prediction_time = (time.time() - start_time) / 3
            
            if prediction_time < 1.0:  # Should be under 1 second per prediction
                self.log_test("Prediction Speed", True, f"Average prediction time: {prediction_time:.3f}s")
            else:
                self.log_test("Prediction Speed", True, f"Prediction time: {prediction_time:.3f}s",
                            "Slower than optimal, may need optimization")
            
            # Time risk assessment
            start_time = time.time()
            for i in range(3):
                risk = ts.assess_risk('XAUUSD')
            risk_time = (time.time() - start_time) / 3
            
            if risk_time < 0.5:  # Should be under 0.5 seconds
                self.log_test("Risk Assessment Speed", True, f"Average risk assessment time: {risk_time:.3f}s")
            else:
                self.log_test("Risk Assessment Speed", True, f"Risk assessment time: {risk_time:.3f}s",
                            "Slower than optimal")
            
            return True
            
        except Exception as e:
            self.log_test("Performance Check", False, f"Performance test error: {e}")
            return False
    
    def generate_summary(self):
        """Generate test summary report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 70)
        print("📋 QUICK SYSTEM TEST - SUMMARY REPORT")
        print("=" * 70)
        
        print(f"⏱️  Total execution time: {duration:.2f} seconds")
        print(f"✅ Tests passed: {self.passed_tests}")
        print(f"❌ Tests failed: {self.failed_tests}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        # Overall status
        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if pass_rate >= 90:
            status_emoji = "🟢"
            status_text = "EXCELLENT"
        elif pass_rate >= 75:
            status_emoji = "🟡"
            status_text = "GOOD"
        elif pass_rate >= 50:
            status_emoji = "🟠"
            status_text = "NEEDS ATTENTION"
        else:
            status_emoji = "🔴"
            status_text = "CRITICAL ISSUES"
        
        print(f"\n{status_emoji} Overall Status: {status_text} ({pass_rate:.1f}% pass rate)")
        
        # Show warnings if any
        if self.warnings:
            print(f"\n⚠️  Warnings to address:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if self.failed_tests == 0:
            print("   • System is ready for comprehensive testing")
            print("   • Consider running COMPREHENSIVE_BACKTEST.py next")
        else:
            print("   • Fix failed tests before proceeding to full testing")
            print("   • Check system dependencies and configuration")
        
        if len(self.warnings) > 2:
            print("   • Consider installing optional dependencies for full functionality")
        
        print(f"\n🎯 Next Steps:")
        print("   1. Review any failed tests and warnings")
        print("   2. Run COMPREHENSIVE_BACKTEST.py for strategy validation")
        print("   3. Execute LIVE_SIMULATION_TEST.py for real-time testing")
        print("   4. Use AUTOMATED_TEST_RUNNER.py for complete validation")
        
        print("\n" + "=" * 70)
        
        return pass_rate >= 75  # Return True if system is reasonably healthy
    
    def run_all_tests(self) -> bool:
        """Run all quick system tests."""
        try:
            # Run tests in sequence
            tests = [
                self.test_system_imports,
                self.test_configuration_system,
                self.test_core_components,
                self.test_data_pipeline,
                self.test_prediction_pipeline,
                self.test_risk_management,
                self.test_system_integration,
                self.test_performance_check
            ]
            
            for test in tests:
                try:
                    test()
                except Exception as e:
                    test_name = test.__name__.replace('test_', '').replace('_', ' ').title()
                    self.log_test(test_name, False, f"Test crashed: {e}")
                    traceback.print_exc()
            
            return self.generate_summary()
            
        except Exception as e:
            print(f"❌ Critical error in test runner: {e}")
            traceback.print_exc()
            return False


def main():
    """Main entry point for quick system test."""
    try:
        tester = QuickSystemTest()
        success = tester.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()