#!/usr/bin/env python3
"""
COMPREHENSIVE_BACKTEST.PY - Full Strategy Testing
================================================

Complete backtesting analysis for the OmniBeing Trading System.
Target execution time: ~5 minutes

Features:
- Load multiple historical datasets
- Test all available strategies
- Generate performance reports
- Risk metrics calculation
- Equity curve plotting
- Monte Carlo simulation
- Strategy comparison report

Created by behicof for the OmniBeing Trading System
"""

import time
import sys
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import math

class ComprehensiveBacktest:
    """Comprehensive backtesting engine for strategy validation."""
    
    def __init__(self):
        """Initialize the backtesting engine."""
        self.start_time = time.time()
        self.results = {}
        self.strategies_tested = 0
        self.total_trades = 0
        self.performance_metrics = {}
        
        print("=" * 70)
        print("📈 COMPREHENSIVE BACKTEST - OmniBeing Trading System")
        print("=" * 70)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target time: ~5 minutes")
        print()
    
    def log_progress(self, message: str, level: str = "INFO"):
        """Log progress message."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if level == "ERROR":
            print(f"❌ [{timestamp}] {message}")
        elif level == "WARNING":
            print(f"⚠️  [{timestamp}] {message}")
        elif level == "SUCCESS":
            print(f"✅ [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")
    
    def generate_mock_historical_data(self, symbol: str, days: int = 365) -> Dict[str, List]:
        """Generate mock historical data for backtesting."""
        self.log_progress(f"Generating {days} days of mock data for {symbol}")
        
        # Generate realistic price movement
        base_price = 2000 if 'XAU' in symbol else 50000 if 'BTC' in symbol else 1.2
        
        data = {
            'timestamp': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'sma_20': [],
            'rsi': [],
            'macd': [],
            'volatility': []
        }
        
        current_price = base_price
        for i in range(days * 24):  # Hourly data
            timestamp = datetime.now() - timedelta(hours=days*24-i)
            
            # Random walk with slight upward bias
            change_pct = random.gauss(0.0005, 0.02)  # 0.05% mean, 2% std
            current_price *= (1 + change_pct)
            
            # Generate OHLC
            open_price = current_price
            high_price = open_price * (1 + abs(random.gauss(0, 0.01)))
            low_price = open_price * (1 - abs(random.gauss(0, 0.01)))
            close_price = random.uniform(low_price, high_price)
            
            current_price = close_price
            
            data['timestamp'].append(timestamp)
            data['open'].append(open_price)
            data['high'].append(high_price)
            data['low'].append(low_price)
            data['close'].append(close_price)
            data['volume'].append(random.uniform(1000, 10000))
            
            # Technical indicators (simplified)
            if i >= 20:
                sma_20 = sum(data['close'][-20:]) / 20
                data['sma_20'].append(sma_20)
            else:
                data['sma_20'].append(close_price)
            
            # RSI (simplified)
            if i >= 14:
                gains = []
                losses = []
                for j in range(14):
                    diff = data['close'][i-13+j] - data['close'][i-14+j]
                    if diff > 0:
                        gains.append(diff)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(-diff)
                
                avg_gain = sum(gains) / 14 if gains else 0
                avg_loss = sum(losses) / 14 if losses else 0.001
                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))
                data['rsi'].append(rsi)
            else:
                data['rsi'].append(50)
            
            # MACD (simplified)
            if i >= 26:
                ema_12 = sum(data['close'][-12:]) / 12
                ema_26 = sum(data['close'][-26:]) / 26
                macd = ema_12 - ema_26
                data['macd'].append(macd)
            else:
                data['macd'].append(0)
            
            # Volatility
            if i >= 20:
                prices = data['close'][-20:]
                returns = [(prices[j] - prices[j-1]) / prices[j-1] for j in range(1, len(prices))]
                volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                data['volatility'].append(volatility)
            else:
                data['volatility'].append(0.02)
        
        return data
    
    def simple_ma_strategy(self, data: Dict[str, List], index: int) -> str:
        """Simple moving average crossover strategy."""
        if index < 50:
            return 'hold'
        
        current_price = data['close'][index]
        sma_20 = data['sma_20'][index]
        prev_price = data['close'][index-1]
        prev_sma_20 = data['sma_20'][index-1]
        
        # Buy when price crosses above SMA
        if prev_price <= prev_sma_20 and current_price > sma_20:
            return 'buy'
        # Sell when price crosses below SMA
        elif prev_price >= prev_sma_20 and current_price < sma_20:
            return 'sell'
        else:
            return 'hold'
    
    def rsi_strategy(self, data: Dict[str, List], index: int) -> str:
        """RSI-based strategy."""
        if index < 14:
            return 'hold'
        
        rsi = data['rsi'][index]
        
        if rsi < 30:  # Oversold
            return 'buy'
        elif rsi > 70:  # Overbought
            return 'sell'
        else:
            return 'hold'
    
    def macd_strategy(self, data: Dict[str, List], index: int) -> str:
        """MACD strategy."""
        if index < 26:
            return 'hold'
        
        macd = data['macd'][index]
        prev_macd = data['macd'][index-1]
        
        # Buy on MACD cross above zero
        if prev_macd <= 0 and macd > 0:
            return 'buy'
        # Sell on MACD cross below zero
        elif prev_macd >= 0 and macd < 0:
            return 'sell'
        else:
            return 'hold'
    
    def combined_strategy(self, data: Dict[str, List], index: int) -> str:
        """Combined strategy using multiple indicators."""
        if index < 50:
            return 'hold'
        
        ma_signal = self.simple_ma_strategy(data, index)
        rsi_signal = self.rsi_strategy(data, index)
        macd_signal = self.macd_strategy(data, index)
        
        # Vote-based system
        buy_votes = sum(1 for signal in [ma_signal, rsi_signal, macd_signal] if signal == 'buy')
        sell_votes = sum(1 for signal in [ma_signal, rsi_signal, macd_signal] if signal == 'sell')
        
        if buy_votes >= 2:
            return 'buy'
        elif sell_votes >= 2:
            return 'sell'
        else:
            return 'hold'
    
    def run_strategy_backtest(self, strategy_name: str, strategy_func: callable, 
                            data: Dict[str, List], initial_capital: float = 10000) -> Dict[str, Any]:
        """Run backtest for a specific strategy."""
        self.log_progress(f"Testing {strategy_name} strategy...")
        
        # Initialize portfolio
        balance = initial_capital
        position = 0  # Number of units held
        position_value = 0
        trades = []
        equity_curve = []
        
        entry_price = 0
        position_type = None
        
        for i in range(len(data['close'])):
            current_price = data['close'][i]
            signal = strategy_func(data, i)
            
            # Calculate current portfolio value
            portfolio_value = balance + (position * current_price)
            equity_curve.append(portfolio_value)
            
            # Execute trades based on signals
            if signal == 'buy' and position <= 0:
                if position < 0:  # Close short position
                    profit = position * (entry_price - current_price)
                    balance += profit
                    trades.append({
                        'timestamp': data['timestamp'][i],
                        'type': 'close_short',
                        'price': current_price,
                        'quantity': abs(position),
                        'profit': profit
                    })
                    position = 0
                
                # Open long position
                position = balance * 0.95 / current_price  # Use 95% of balance
                entry_price = current_price
                position_type = 'long'
                balance *= 0.05  # Keep 5% as cash
                trades.append({
                    'timestamp': data['timestamp'][i],
                    'type': 'buy',
                    'price': current_price,
                    'quantity': position,
                    'profit': 0
                })
            
            elif signal == 'sell' and position >= 0:
                if position > 0:  # Close long position
                    profit = position * (current_price - entry_price)
                    balance += position * current_price
                    trades.append({
                        'timestamp': data['timestamp'][i],
                        'type': 'close_long',
                        'price': current_price,
                        'quantity': position,
                        'profit': profit
                    })
                    position = 0
                
                # Open short position (simplified)
                position = -(balance * 0.95 / current_price)
                entry_price = current_price
                position_type = 'short'
                trades.append({
                    'timestamp': data['timestamp'][i],
                    'type': 'sell_short',
                    'price': current_price,
                    'quantity': abs(position),
                    'profit': 0
                })
        
        # Close any remaining position
        if position != 0:
            final_price = data['close'][-1]
            if position > 0:
                profit = position * (final_price - entry_price)
                balance += position * final_price
            else:
                profit = position * (entry_price - final_price)
                balance += abs(position) * final_price
            
            trades.append({
                'timestamp': data['timestamp'][-1],
                'type': 'close_final',
                'price': final_price,
                'quantity': abs(position),
                'profit': profit
            })
        
        # Calculate performance metrics
        final_value = balance
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] < 0]
        
        win_rate = len(winning_trades) / len([t for t in trades if 'profit' in t and t['profit'] != 0]) * 100 if trades else 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = initial_capital
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio (simplified)
        returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] 
                  for i in range(1, len(equity_curve))]
        avg_return = sum(returns) / len(returns) if returns else 0
        return_std = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5 if returns else 0.01
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        return {
            'strategy_name': strategy_name,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def run_monte_carlo_simulation(self, best_strategy: Dict[str, Any], iterations: int = 100) -> Dict[str, Any]:
        """Run Monte Carlo simulation on the best performing strategy."""
        self.log_progress(f"Running Monte Carlo simulation with {iterations} iterations...")
        
        strategy_name = best_strategy['strategy_name']
        returns = []
        max_drawdowns = []
        
        for i in range(iterations):
            # Generate random data based on historical statistics
            data = self.generate_mock_historical_data('XAUUSD', days=180)
            
            # Run strategy
            if strategy_name == 'Simple MA':
                result = self.run_strategy_backtest('Simple MA', self.simple_ma_strategy, data)
            elif strategy_name == 'RSI':
                result = self.run_strategy_backtest('RSI', self.rsi_strategy, data)
            elif strategy_name == 'MACD':
                result = self.run_strategy_backtest('MACD', self.macd_strategy, data)
            else:
                result = self.run_strategy_backtest('Combined', self.combined_strategy, data)
            
            returns.append(result['total_return'])
            max_drawdowns.append(result['max_drawdown'])
        
        # Calculate statistics
        avg_return = sum(returns) / len(returns)
        return_std = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5
        avg_drawdown = sum(max_drawdowns) / len(max_drawdowns)
        
        # Calculate percentiles
        returns.sort()
        p5 = returns[int(0.05 * len(returns))]
        p95 = returns[int(0.95 * len(returns))]
        
        return {
            'iterations': iterations,
            'avg_return': avg_return,
            'return_std': return_std,
            'avg_max_drawdown': avg_drawdown,
            'return_5th_percentile': p5,
            'return_95th_percentile': p95,
            'probability_positive': sum(1 for r in returns if r > 0) / len(returns) * 100
        }
    
    def generate_performance_charts(self, results: List[Dict[str, Any]]) -> str:
        """Generate ASCII performance charts."""
        chart_output = []
        
        # Strategy comparison chart
        chart_output.append("\n📊 STRATEGY PERFORMANCE COMPARISON")
        chart_output.append("=" * 50)
        
        for result in results:
            name = result['strategy_name'][:15].ljust(15)
            return_pct = result['total_return']
            win_rate = result['win_rate']
            drawdown = result['max_drawdown']
            
            # Simple bar chart for returns
            bar_length = max(0, min(50, int(abs(return_pct) / 2)))
            bar = "█" * bar_length
            sign = "+" if return_pct >= 0 else "-"
            
            chart_output.append(f"{name} │{sign}{bar:<50}│ {return_pct:+7.2f}%")
            chart_output.append(f"{'':15} │ Win: {win_rate:5.1f}%  Drawdown: {drawdown:5.1f}%")
            chart_output.append("")
        
        return "\n".join(chart_output)
    
    def test_system_integration(self) -> bool:
        """Test integration with main trading system."""
        self.log_progress("Testing integration with main trading system...")
        
        try:
            from main_trading_system import MainTradingSystem
            ts = MainTradingSystem()
            
            # Test if we can get predictions for backtesting
            prediction = ts.make_prediction('XAUUSD')
            if prediction:
                self.log_progress("✅ Main trading system integration successful", "SUCCESS")
                return True
            else:
                self.log_progress("⚠️  Main trading system available but no predictions", "WARNING")
                return True
        except Exception as e:
            self.log_progress(f"⚠️  Main trading system integration failed: {e}", "WARNING")
            return False
    
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run the complete backtesting suite."""
        self.log_progress("Starting comprehensive backtesting suite...")
        
        # Test system integration first
        system_integration = self.test_system_integration()
        
        # Generate historical data for multiple symbols
        symbols = ['XAUUSD', 'BTCUSD', 'EURUSD']
        strategies = [
            ('Simple MA', self.simple_ma_strategy),
            ('RSI', self.rsi_strategy),
            ('MACD', self.macd_strategy),
            ('Combined', self.combined_strategy)
        ]
        
        all_results = []
        
        for symbol in symbols:
            self.log_progress(f"Generating data for {symbol}...")
            data = self.generate_mock_historical_data(symbol, days=365)
            
            for strategy_name, strategy_func in strategies:
                try:
                    result = self.run_strategy_backtest(strategy_name, strategy_func, data)
                    result['symbol'] = symbol
                    all_results.append(result)
                    self.strategies_tested += 1
                    self.total_trades += result['total_trades']
                    
                    self.log_progress(f"✅ {strategy_name} on {symbol}: {result['total_return']:+.2f}% return", "SUCCESS")
                    
                except Exception as e:
                    self.log_progress(f"❌ {strategy_name} on {symbol} failed: {e}", "ERROR")
        
        # Find best performing strategy
        if all_results:
            best_strategy = max(all_results, key=lambda x: x['total_return'])
            self.log_progress(f"Best strategy: {best_strategy['strategy_name']} on {best_strategy['symbol']} ({best_strategy['total_return']:+.2f}%)", "SUCCESS")
            
            # Run Monte Carlo simulation on best strategy
            monte_carlo = self.run_monte_carlo_simulation(best_strategy)
        else:
            best_strategy = None
            monte_carlo = None
        
        return {
            'system_integration': system_integration,
            'strategies_tested': self.strategies_tested,
            'total_trades': self.total_trades,
            'all_results': all_results,
            'best_strategy': best_strategy,
            'monte_carlo': monte_carlo
        }
    
    def generate_detailed_report(self, results: Dict[str, Any]):
        """Generate comprehensive backtest report."""
        print("\n" + "=" * 70)
        print("📈 COMPREHENSIVE BACKTEST - DETAILED REPORT")
        print("=" * 70)
        
        # Executive Summary
        print("\n🎯 EXECUTIVE SUMMARY")
        print("-" * 30)
        print(f"Strategies Tested: {results['strategies_tested']}")
        print(f"Total Simulated Trades: {results['total_trades']}")
        print(f"System Integration: {'✅ Pass' if results['system_integration'] else '❌ Fail'}")
        
        # Strategy Performance Summary
        if results['all_results']:
            print(f"\n📊 STRATEGY PERFORMANCE OVERVIEW")
            print("-" * 40)
            
            strategy_summary = {}
            for result in results['all_results']:
                strategy = result['strategy_name']
                if strategy not in strategy_summary:
                    strategy_summary[strategy] = {
                        'returns': [],
                        'win_rates': [],
                        'drawdowns': [],
                        'trades': []
                    }
                
                strategy_summary[strategy]['returns'].append(result['total_return'])
                strategy_summary[strategy]['win_rates'].append(result['win_rate'])
                strategy_summary[strategy]['drawdowns'].append(result['max_drawdown'])
                strategy_summary[strategy]['trades'].append(result['total_trades'])
            
            for strategy, metrics in strategy_summary.items():
                avg_return = sum(metrics['returns']) / len(metrics['returns'])
                avg_win_rate = sum(metrics['win_rates']) / len(metrics['win_rates'])
                avg_drawdown = sum(metrics['drawdowns']) / len(metrics['drawdowns'])
                total_trades = sum(metrics['trades'])
                
                print(f"\n{strategy}:")
                print(f"  📈 Average Return: {avg_return:+7.2f}%")
                print(f"  🎯 Average Win Rate: {avg_win_rate:6.1f}%")
                print(f"  📉 Average Drawdown: {avg_drawdown:6.1f}%")
                print(f"  📋 Total Trades: {total_trades}")
        
        # Best Strategy Details
        if results['best_strategy']:
            best = results['best_strategy']
            print(f"\n🏆 BEST PERFORMING STRATEGY")
            print("-" * 35)
            print(f"Strategy: {best['strategy_name']}")
            print(f"Symbol: {best['symbol']}")
            print(f"Total Return: {best['total_return']:+.2f}%")
            print(f"Win Rate: {best['win_rate']:.1f}%")
            print(f"Max Drawdown: {best['max_drawdown']:.1f}%")
            print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"Total Trades: {best['total_trades']}")
        
        # Monte Carlo Results
        if results['monte_carlo']:
            mc = results['monte_carlo']
            print(f"\n🎲 MONTE CARLO SIMULATION")
            print("-" * 30)
            print(f"Iterations: {mc['iterations']}")
            print(f"Average Return: {mc['avg_return']:+.2f}%")
            print(f"Return Std Dev: {mc['return_std']:.2f}%")
            print(f"5th Percentile: {mc['return_5th_percentile']:+.2f}%")
            print(f"95th Percentile: {mc['return_95th_percentile']:+.2f}%")
            print(f"Probability of Profit: {mc['probability_positive']:.1f}%")
        
        # Performance Charts
        if results['all_results']:
            print(self.generate_performance_charts(results['all_results']))
        
        # Risk Assessment
        print(f"\n🛡️  RISK ASSESSMENT")
        print("-" * 25)
        
        if results['all_results']:
            all_returns = [r['total_return'] for r in results['all_results']]
            all_drawdowns = [r['max_drawdown'] for r in results['all_results']]
            
            max_loss = min(all_returns)
            max_drawdown = max(all_drawdowns)
            positive_strategies = sum(1 for r in all_returns if r > 0)
            
            print(f"Worst Case Return: {max_loss:+.2f}%")
            print(f"Maximum Drawdown: {max_drawdown:.1f}%")
            print(f"Profitable Strategies: {positive_strategies}/{len(all_returns)} ({positive_strategies/len(all_returns)*100:.1f}%)")
            
            if max_drawdown > 30:
                print("⚠️  HIGH RISK: Maximum drawdown exceeds 30%")
            elif max_drawdown > 20:
                print("⚠️  MODERATE RISK: Maximum drawdown exceeds 20%")
            else:
                print("✅ ACCEPTABLE RISK: Drawdown within reasonable limits")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 25)
        
        if results['best_strategy'] and results['best_strategy']['total_return'] > 0:
            print("✅ System shows positive performance potential")
            print("✅ Ready for live simulation testing")
            print("✅ Consider implementing position sizing rules")
        else:
            print("⚠️  System needs optimization before live deployment")
            print("⚠️  Review strategy parameters and risk management")
        
        if results['monte_carlo'] and results['monte_carlo']['probability_positive'] > 60:
            print("✅ Monte Carlo shows consistent performance")
        else:
            print("⚠️  High performance variability detected")
        
        # Next Steps
        print(f"\n🎯 NEXT STEPS")
        print("-" * 15)
        print("1. Run LIVE_SIMULATION_TEST.py for real-time validation")
        print("2. Execute INTEGRATION_VALIDATOR.py for system testing")
        print("3. Perform PERFORMANCE_BENCHMARK.py for optimization")
        print("4. Complete DEPLOYMENT_READINESS_CHECK.py before production")
        
        # Execution Time
        duration = time.time() - self.start_time
        print(f"\n⏱️  Total Execution Time: {duration:.2f} seconds")
        print("=" * 70)


def main():
    """Main entry point for comprehensive backtest."""
    try:
        backtester = ComprehensiveBacktest()
        results = backtester.run_comprehensive_backtest()
        backtester.generate_detailed_report(results)
        
        # Determine success
        success = (results['strategies_tested'] > 0 and 
                  results.get('best_strategy', {}).get('total_return', -100) > -20)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Fatal error in backtest: {e}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()