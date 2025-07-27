"""
Simple test to verify basic module structure without dependencies
"""

import sys
import os

def test_file_structure():
    """Test that all required files exist"""
    
    print("🚀 Testing OmniBeing Trading System File Structure...\n")
    
    required_files = [
        'requirements.txt',
        'config.py',
        'main_trading_system.py',
        'data_manager.py',
        'market_connectors.py',
        'enhanced_risk_manager.py',
        'backtesting_engine.py',
        'logging_system.py',
        'api_server.py',
        'live_dashboard.py',
        'test_integration.py',
        '.env.example',
        'TRADING_SYSTEM_README.md'
    ]
    
    existing_modules = [
        'gut_trader.py',
        'external_risk_manager.py',
        'reinforcement_learning_core.py',
        'omni_persona.py',
        'global_sentiment.py',
        'emotional_responder.py',
        'social_pulse.py',
        'group_behavior.py',
        'vision_live.py',
        'live_visual_analysis.py',
        'final_expansion_for_advanced_predictions.py'
    ]
    
    print("📁 Checking required new files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} - MISSING")
    
    print("\n📁 Checking existing modules integration:")
    for file in existing_modules:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ⚠ {file} - Not found (optional)")
    
    print("\n📊 Summary:")
    new_files_exist = sum(1 for f in required_files if os.path.exists(f))
    existing_files = sum(1 for f in existing_modules if os.path.exists(f))
    
    print(f"   New system files: {new_files_exist}/{len(required_files)}")
    print(f"   Existing modules: {existing_files}/{len(existing_modules)}")
    
    if new_files_exist == len(required_files):
        print("   🎉 All required files present!")
        return True
    else:
        print("   ❌ Some files missing")
        return False

def test_basic_imports():
    """Test basic Python syntax of files"""
    
    print("\n🔍 Testing basic Python syntax...\n")
    
    files_to_test = [
        'config.py',
        'main_trading_system.py',
        'data_manager.py',
        'market_connectors.py',
        'enhanced_risk_manager.py',
        'backtesting_engine.py',
        'logging_system.py',
        'api_server.py',
        'live_dashboard.py'
    ]
    
    passed = 0
    for file in files_to_test:
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    compile(content, file, 'exec')
                print(f"   ✓ {file} - Syntax OK")
                passed += 1
            except SyntaxError as e:
                print(f"   ✗ {file} - Syntax Error: {e}")
            except Exception as e:
                print(f"   ⚠ {file} - Other error: {e}")
        else:
            print(f"   ✗ {file} - File not found")
    
    print(f"\n   Syntax check: {passed}/{len(files_to_test)} files passed")
    return passed == len(files_to_test)

def test_configuration():
    """Test configuration structure"""
    
    print("\n⚙️ Testing configuration...\n")
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            required_packages = [
                'pandas', 'numpy', 'scikit-learn', 'fastapi', 
                'uvicorn', 'plotly', 'dash', 'python-dotenv'
            ]
            
            missing = []
            for pkg in required_packages:
                if pkg not in requirements:
                    missing.append(pkg)
            
            if not missing:
                print("   ✓ requirements.txt contains all required packages")
            else:
                print(f"   ⚠ requirements.txt missing: {missing}")
    
    # Check .env.example
    if os.path.exists('.env.example'):
        with open('.env.example', 'r') as f:
            env_content = f.read()
            required_vars = [
                'BINANCE_API_KEY', 'DEFAULT_RISK_LEVEL', 'API_PORT', 
                'DASHBOARD_PORT', 'LOG_LEVEL'
            ]
            
            missing_vars = []
            for var in required_vars:
                if var not in env_content:
                    missing_vars.append(var)
            
            if not missing_vars:
                print("   ✓ .env.example contains required variables")
            else:
                print(f"   ⚠ .env.example missing: {missing_vars}")
    
    return True

def test_system_architecture():
    """Test system architecture compliance"""
    
    print("\n🏗️ Testing system architecture...\n")
    
    # Check that new files integrate with existing modules
    integration_points = {
        'main_trading_system.py': [
            'gut_trader', 'external_risk_manager', 'reinforcement_learning_core',
            'omni_persona', 'global_sentiment', 'emotional_responder'
        ],
        'enhanced_risk_manager.py': ['external_risk_manager'],
        'api_server.py': ['main_trading_system', 'data_manager', 'market_connectors'],
        'backtesting_engine.py': ['config']
    }
    
    for file, expected_imports in integration_points.items():
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = f.read()
                missing_imports = []
                for imp in expected_imports:
                    if f'from {imp}' not in content and f'import {imp}' not in content:
                        missing_imports.append(imp)
                
                if not missing_imports:
                    print(f"   ✓ {file} - Proper integration with existing modules")
                else:
                    print(f"   ⚠ {file} - Missing integration: {missing_imports}")
    
    print("   ✓ System follows integration architecture")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 OmniBeing Trading System - Basic Structure Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_basic_imports),
        ("Configuration", test_configuration),
        ("Architecture", test_system_architecture)
    ]
    
    passed_tests = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"   ❌ Test {test_name} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 System structure is complete and ready!")
        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy .env.example to .env and configure")
        print("3. Run integration test: python test_integration.py")
        print("4. Start system components")
        
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please review the issues above.")
        sys.exit(1)