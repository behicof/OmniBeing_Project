#!/usr/bin/env python3
"""
AUTOMATED_TEST_RUNNER.PY - Test Orchestration
=============================================

Automated test execution for the OmniBeing Trading System.
Runs all test suites and generates comprehensive reports.

Features:
- Run all test suites
- Generate comprehensive reports
- Performance summaries
- Success/failure tracking
- Test result visualization
- Automated reporting

Created by behicof for the OmniBeing Trading System
"""

import time
import sys
import os
import json
import subprocess
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading

class AutomatedTestRunner:
    """Automated test runner for the complete OmniBeing Trading System."""
    
    def __init__(self):
        """Initialize the automated test runner."""
        self.start_time = time.time()
        self.test_suite_results = {}
        self.overall_metrics = {}
        self.execution_times = {}
        self.total_tests_run = 0
        self.total_tests_passed = 0
        self.total_warnings = 0
        
        print("=" * 80)
        print("🤖 AUTOMATED TEST RUNNER - OmniBeing Trading System")
        print("=" * 80)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Running complete test suite validation")
        print()
    
    def log_progress(self, message: str, level: str = "INFO"):
        """Log progress message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if level == "ERROR":
            print(f"❌ [{timestamp}] {message}")
        elif level == "WARNING":
            print(f"⚠️  [{timestamp}] {message}")
        elif level == "SUCCESS":
            print(f"✅ [{timestamp}] {message}")
        elif level == "PHASE":
            print(f"\n🔄 [{timestamp}] {message}")
            print("-" * 60)
        else:
            print(f"ℹ️  [{timestamp}] {message}")
    
    def run_test_script(self, script_name: str, description: str, timeout: int = 300) -> Dict[str, Any]:
        """Run a test script and capture results."""
        self.log_progress(f"Starting {description}...", "PHASE")
        
        script_path = f"./{script_name}"
        if not os.path.exists(script_path):
            return {
                'success': False,
                'error': f"Script {script_name} not found",
                'execution_time': 0,
                'output': '',
                'exit_code': -1
            }
        
        start_time = time.time()
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            execution_time = time.time() - start_time
            
            # Parse output for metrics
            output_lines = result.stdout.split('\n') if result.stdout else []
            
            # Extract basic metrics from output
            tests_passed = 0
            tests_failed = 0
            warnings = 0
            
            for line in output_lines:
                if '✅ PASS' in line or '✅' in line:
                    tests_passed += 1
                elif '❌ FAIL' in line or '❌' in line:
                    tests_failed += 1
                elif '⚠️' in line or 'WARNING' in line:
                    warnings += 1
            
            success = result.returncode == 0
            
            if success:
                self.log_progress(f"✅ {description} completed successfully in {execution_time:.1f}s", "SUCCESS")
            else:
                self.log_progress(f"❌ {description} failed (exit code: {result.returncode})", "ERROR")
            
            return {
                'success': success,
                'execution_time': execution_time,
                'output': result.stdout,
                'error_output': result.stderr,
                'exit_code': result.returncode,
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'warnings': warnings
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.log_progress(f"⏰ {description} timed out after {execution_time:.1f}s", "WARNING")
            
            return {
                'success': False,
                'error': f"Timeout after {timeout}s",
                'execution_time': execution_time,
                'output': 'Test timed out',
                'exit_code': -2
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_progress(f"💥 {description} crashed: {e}", "ERROR")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'output': '',
                'exit_code': -3
            }
    
    def run_test_phase_1_quick_validation(self) -> Dict[str, Any]:
        """Phase 1: Quick system validation (2 minutes)."""
        self.log_progress("PHASE 1: Quick System Validation", "PHASE")
        
        results = {}
        
        # Quick System Test
        quick_test_result = self.run_test_script(
            'QUICK_SYSTEM_TEST.py',
            'Quick System Test - Rapid system validation',
            timeout=180  # 3 minutes
        )
        results['quick_system_test'] = quick_test_result
        
        # Update metrics
        if quick_test_result['success']:
            self.total_tests_passed += quick_test_result.get('tests_passed', 0)
        
        self.total_tests_run += (quick_test_result.get('tests_passed', 0) + 
                               quick_test_result.get('tests_failed', 0))
        self.total_warnings += quick_test_result.get('warnings', 0)
        
        # Phase summary
        phase_success = quick_test_result['success']
        phase_time = quick_test_result['execution_time']
        
        if phase_success:
            self.log_progress(f"Phase 1 completed successfully in {phase_time:.1f}s", "SUCCESS")
        else:
            self.log_progress(f"Phase 1 had issues - continuing with remaining tests", "WARNING")
        
        return {
            'phase_success': phase_success,
            'phase_time': phase_time,
            'results': results
        }
    
    def run_test_phase_2_core_testing(self) -> Dict[str, Any]:
        """Phase 2: Core testing (8 minutes)."""
        self.log_progress("PHASE 2: Core System Testing", "PHASE")
        
        results = {}
        phase_start = time.time()
        
        # Comprehensive Backtest
        backtest_result = self.run_test_script(
            'COMPREHENSIVE_BACKTEST.py',
            'Comprehensive Backtest - Strategy validation',
            timeout=360  # 6 minutes
        )
        results['comprehensive_backtest'] = backtest_result
        
        # Live Simulation Test
        simulation_result = self.run_test_script(
            'LIVE_SIMULATION_TEST.py',
            'Live Simulation Test - Real-time testing',
            timeout=240  # 4 minutes
        )
        results['live_simulation_test'] = simulation_result
        
        # Update metrics
        for result in [backtest_result, simulation_result]:
            if result['success']:
                self.total_tests_passed += result.get('tests_passed', 0)
            
            self.total_tests_run += (result.get('tests_passed', 0) + 
                                   result.get('tests_failed', 0))
            self.total_warnings += result.get('warnings', 0)
        
        # Phase summary
        phase_time = time.time() - phase_start
        successful_tests = sum(1 for r in results.values() if r['success'])
        phase_success = successful_tests >= len(results) // 2  # At least half successful
        
        if phase_success:
            self.log_progress(f"Phase 2 completed - {successful_tests}/{len(results)} tests successful in {phase_time:.1f}s", "SUCCESS")
        else:
            self.log_progress(f"Phase 2 had significant issues - {successful_tests}/{len(results)} tests successful", "WARNING")
        
        return {
            'phase_success': phase_success,
            'phase_time': phase_time,
            'results': results
        }
    
    def run_test_phase_3_integration_testing(self) -> Dict[str, Any]:
        """Phase 3: Integration testing (3 minutes)."""
        self.log_progress("PHASE 3: Integration Testing", "PHASE")
        
        results = {}
        phase_start = time.time()
        
        # Integration Validator
        integration_result = self.run_test_script(
            'INTEGRATION_VALIDATOR.py',
            'Integration Validator - Full system integration',
            timeout=300  # 5 minutes
        )
        results['integration_validator'] = integration_result
        
        # Update metrics
        if integration_result['success']:
            self.total_tests_passed += integration_result.get('tests_passed', 0)
        
        self.total_tests_run += (integration_result.get('tests_passed', 0) + 
                               integration_result.get('tests_failed', 0))
        self.total_warnings += integration_result.get('warnings', 0)
        
        # Phase summary
        phase_time = time.time() - phase_start
        phase_success = integration_result['success']
        
        if phase_success:
            self.log_progress(f"Phase 3 completed successfully in {phase_time:.1f}s", "SUCCESS")
        else:
            self.log_progress(f"Phase 3 identified integration issues", "WARNING")
        
        return {
            'phase_success': phase_success,
            'phase_time': phase_time,
            'results': results
        }
    
    def run_test_phase_4_performance_and_production(self) -> Dict[str, Any]:
        """Phase 4: Performance & production readiness (10 minutes)."""
        self.log_progress("PHASE 4: Performance & Production Readiness", "PHASE")
        
        results = {}
        phase_start = time.time()
        
        # Performance Benchmark
        performance_result = self.run_test_script(
            'PERFORMANCE_BENCHMARK.py',
            'Performance Benchmark - System performance analysis',
            timeout=420  # 7 minutes
        )
        results['performance_benchmark'] = performance_result
        
        # Deployment Readiness Check
        deployment_result = self.run_test_script(
            'DEPLOYMENT_READINESS_CHECK.py',
            'Deployment Readiness Check - Production validation',
            timeout=360  # 6 minutes
        )
        results['deployment_readiness_check'] = deployment_result
        
        # Update metrics
        for result in [performance_result, deployment_result]:
            if result['success']:
                self.total_tests_passed += result.get('tests_passed', 0)
            
            self.total_tests_run += (result.get('tests_passed', 0) + 
                                   result.get('tests_failed', 0))
            self.total_warnings += result.get('warnings', 0)
        
        # Phase summary
        phase_time = time.time() - phase_start
        successful_tests = sum(1 for r in results.values() if r['success'])
        phase_success = successful_tests >= len(results) // 2
        
        if phase_success:
            self.log_progress(f"Phase 4 completed - {successful_tests}/{len(results)} tests successful in {phase_time:.1f}s", "SUCCESS")
        else:
            self.log_progress(f"Phase 4 had issues - {successful_tests}/{len(results)} tests successful", "WARNING")
        
        return {
            'phase_success': phase_success,
            'phase_time': phase_time,
            'results': results
        }
    
    def run_parallel_tests(self, test_configs: List[Tuple[str, str, int]]) -> Dict[str, Any]:
        """Run multiple tests in parallel for efficiency."""
        self.log_progress("Running tests in parallel for efficiency...", "INFO")
        
        results = {}
        threads = []
        thread_results = {}
        
        def run_test_thread(script_name: str, description: str, timeout: int):
            result = self.run_test_script(script_name, description, timeout)
            thread_results[script_name] = result
        
        # Start all threads
        for script_name, description, timeout in test_configs:
            thread = threading.Thread(
                target=run_test_thread,
                args=(script_name, description, timeout)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return thread_results
    
    def generate_test_summary(self, phase_results: Dict[str, Any]) -> str:
        """Generate ASCII visualization of test results."""
        summary_lines = []
        
        # Test results overview
        summary_lines.append("\n📊 TEST EXECUTION SUMMARY")
        summary_lines.append("=" * 40)
        
        total_execution_time = sum(phase['phase_time'] for phase in phase_results.values())
        successful_phases = sum(1 for phase in phase_results.values() if phase['phase_success'])
        
        summary_lines.append(f"⏱️  Total Execution Time: {total_execution_time:.1f} seconds")
        summary_lines.append(f"🏆 Successful Phases: {successful_phases}/{len(phase_results)}")
        summary_lines.append(f"📋 Total Tests Run: {self.total_tests_run}")
        summary_lines.append(f"✅ Tests Passed: {self.total_tests_passed}")
        summary_lines.append(f"⚠️  Total Warnings: {self.total_warnings}")
        
        # Phase-by-phase breakdown
        summary_lines.append(f"\n📈 PHASE BREAKDOWN")
        summary_lines.append("-" * 30)
        
        phase_names = {
            'phase_1': 'Quick Validation',
            'phase_2': 'Core Testing', 
            'phase_3': 'Integration Testing',
            'phase_4': 'Performance & Production'
        }
        
        for phase_key, phase_data in phase_results.items():
            phase_name = phase_names.get(phase_key, phase_key)
            status = "✅" if phase_data['phase_success'] else "❌"
            time_str = f"{phase_data['phase_time']:.1f}s"
            
            summary_lines.append(f"{status} {phase_name:<25} {time_str:>8}")
            
            # Show individual test results for this phase
            if 'results' in phase_data:
                for test_name, test_result in phase_data['results'].items():
                    test_status = "  ✅" if test_result['success'] else "  ❌"
                    test_time = f"{test_result['execution_time']:.1f}s"
                    test_display_name = test_name.replace('_', ' ').title()
                    summary_lines.append(f"{test_status} {test_display_name:<23} {test_time:>8}")
        
        return "\n".join(summary_lines)
    
    def generate_comprehensive_report(self, phase_results: Dict[str, Any]):
        """Generate the final comprehensive test report."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("🤖 AUTOMATED TEST RUNNER - COMPREHENSIVE REPORT")
        print("=" * 80)
        
        # Executive Summary
        print(f"\n🎯 EXECUTIVE SUMMARY")
        print("-" * 30)
        print(f"📅 Test Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total Runtime: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"📊 Test Phases: {len(phase_results)}")
        print(f"🔧 Individual Tests: {len([t for phase in phase_results.values() for t in phase.get('results', {})])}")
        
        # Overall System Health
        successful_phases = sum(1 for phase in phase_results.values() if phase['phase_success'])
        phase_success_rate = (successful_phases / len(phase_results)) * 100
        
        if self.total_tests_run > 0:
            test_success_rate = (self.total_tests_passed / self.total_tests_run) * 100
        else:
            test_success_rate = 0
        
        if phase_success_rate >= 90 and test_success_rate >= 85:
            health_status = "🟢 EXCELLENT"
            health_desc = "System is in excellent condition"
        elif phase_success_rate >= 75 and test_success_rate >= 70:
            health_status = "🟡 GOOD"
            health_desc = "System is functional with minor issues"
        elif phase_success_rate >= 50 and test_success_rate >= 50:
            health_status = "🟠 NEEDS ATTENTION"
            health_desc = "System has issues requiring attention"
        else:
            health_status = "🔴 CRITICAL"
            health_desc = "System has serious issues"
        
        print(f"\n{health_status}")
        print(f"📈 Phase Success Rate: {phase_success_rate:.1f}%")
        print(f"✅ Test Success Rate: {test_success_rate:.1f}%")
        print(f"💬 {health_desc}")
        
        # Test Summary Visualization
        print(self.generate_test_summary(phase_results))
        
        # Detailed Phase Analysis
        print(f"\n🔍 DETAILED PHASE ANALYSIS")
        print("-" * 35)
        
        for phase_key, phase_data in phase_results.items():
            phase_name = phase_key.replace('_', ' ').title()
            print(f"\n📋 {phase_name}")
            print(f"   Status: {'✅ PASSED' if phase_data['phase_success'] else '❌ FAILED'}")
            print(f"   Duration: {phase_data['phase_time']:.1f} seconds")
            
            if 'results' in phase_data:
                results = phase_data['results']
                successful_tests = sum(1 for r in results.values() if r['success'])
                print(f"   Tests: {successful_tests}/{len(results)} successful")
                
                # Show failed tests
                failed_tests = [name for name, result in results.items() if not result['success']]
                if failed_tests:
                    print(f"   Failed: {', '.join(failed_tests)}")
        
        # Performance Metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print("-" * 30)
        
        fastest_phase = min(phase_results.values(), key=lambda x: x['phase_time'])
        slowest_phase = max(phase_results.values(), key=lambda x: x['phase_time'])
        
        print(f"Fastest Phase: {fastest_phase['phase_time']:.1f}s")
        print(f"Slowest Phase: {slowest_phase['phase_time']:.1f}s")
        print(f"Average Phase Time: {sum(p['phase_time'] for p in phase_results.values()) / len(phase_results):.1f}s")
        
        if self.total_tests_run > 0:
            avg_test_time = total_duration / self.total_tests_run
            print(f"Average Test Time: {avg_test_time:.2f}s")
        
        # Warnings and Issues
        if self.total_warnings > 0:
            print(f"\n⚠️  WARNINGS AND ISSUES")
            print("-" * 30)
            print(f"Total Warnings: {self.total_warnings}")
            print("Review individual test outputs for detailed warning information.")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 25)
        
        recommendations = []
        
        if phase_success_rate < 100:
            recommendations.append("Address failed test phases before production deployment")
        
        if test_success_rate < 90:
            recommendations.append("Investigate and fix failing individual tests")
        
        if self.total_warnings > 5:
            recommendations.append("Review and address system warnings")
        
        if total_duration > 1200:  # 20 minutes
            recommendations.append("Consider optimizing test execution time")
        
        # Default recommendations
        recommendations.extend([
            "Monitor system performance in production",
            "Set up automated testing in CI/CD pipeline",
            "Regularly run test suite to catch regressions",
            "Update test coverage as system evolves"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Next Steps
        print(f"\n🎯 NEXT STEPS")
        print("-" * 15)
        
        if phase_success_rate >= 90 and test_success_rate >= 85:
            print("🟢 SYSTEM READY:")
            print("  • All major test phases passed")
            print("  • System is ready for production deployment")
            print("  • Continue with deployment procedures")
            print("  • Set up production monitoring")
        elif phase_success_rate >= 70:
            print("🟡 SYSTEM NEEDS REVIEW:")
            print("  • Some test phases failed or had issues")
            print("  • Review failed tests and address issues")
            print("  • Re-run specific test phases after fixes")
            print("  • Consider deployment with close monitoring")
        else:
            print("🔴 SYSTEM NOT READY:")
            print("  • Multiple test phases failed")
            print("  • Significant issues detected")
            print("  • Address all critical issues before deployment")
            print("  • Re-run complete test suite after fixes")
        
        # Test Report Storage
        print(f"\n📄 TEST REPORT STORAGE")
        print("-" * 30)
        
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_duration,
                'phase_results': phase_results,
                'overall_metrics': {
                    'total_tests_run': self.total_tests_run,
                    'total_tests_passed': self.total_tests_passed,
                    'total_warnings': self.total_warnings,
                    'phase_success_rate': phase_success_rate,
                    'test_success_rate': test_success_rate
                }
            }
            
            report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"✅ Test report saved to: {report_filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save test report: {e}")
        
        print(f"\n⏱️  Report generated in {time.time() - end_time:.2f} seconds")
        print("=" * 80)
        
        return phase_success_rate >= 70
    
    def run_complete_test_suite(self) -> bool:
        """Run the complete automated test suite."""
        self.log_progress("Starting automated test suite execution", "INFO")
        self.log_progress("This will run all test phases in sequence", "INFO")
        
        phase_results = {}
        
        try:
            # Phase 1: Quick Validation
            phase_results['phase_1'] = self.run_test_phase_1_quick_validation()
            
            # Phase 2: Core Testing  
            phase_results['phase_2'] = self.run_test_phase_2_core_testing()
            
            # Phase 3: Integration Testing
            phase_results['phase_3'] = self.run_test_phase_3_integration_testing()
            
            # Phase 4: Performance & Production
            phase_results['phase_4'] = self.run_test_phase_4_performance_and_production()
            
            # Generate comprehensive report
            success = self.generate_comprehensive_report(phase_results)
            
            return success
            
        except KeyboardInterrupt:
            self.log_progress("Test suite interrupted by user", "WARNING")
            print("\n⚠️  Test execution was interrupted")
            print("Partial results may be available")
            return False
            
        except Exception as e:
            self.log_progress(f"Critical error in test suite: {e}", "ERROR")
            traceback.print_exc()
            return False


def main():
    """Main entry point for automated test runner."""
    try:
        print("🤖 OmniBeing Trading System - Automated Test Runner")
        print("   Complete validation suite for production readiness")
        print()
        
        runner = AutomatedTestRunner()
        success = runner.run_complete_test_suite()
        
        if success:
            print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("   System validation complete - ready for deployment")
        else:
            print("\n⚠️  TEST SUITE COMPLETED WITH ISSUES")
            print("   Review test results and address issues before deployment")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Automated test runner interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Fatal error in test runner: {e}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()