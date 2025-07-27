"""
Backtesting Engine
Strategy testing with historical data, performance metrics, and optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import itertools
from concurrent.futures import ProcessPoolExecutor
import json

from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtesting results data structure"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    parameters: Dict[str, Any]
    trade_log: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]

@dataclass
class Trade:
    """Individual trade data structure"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    status: str  # 'open', 'closed'
    reason: str  # 'signal', 'stop_loss', 'take_profit', 'time_exit'

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on data"""
        raise NotImplementedError
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        return data

class MovingAverageCrossStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        super().__init__("MA_Cross", {
            'fast_period': fast_period,
            'slow_period': slow_period
        })
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        data = data.copy()
        data['ma_fast'] = data['close'].rolling(self.fast_period).mean()
        data['ma_slow'] = data['close'].rolling(self.slow_period).mean()
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MA crossover"""
        data = self.calculate_indicators(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: fast MA crosses above slow MA
        buy_signals = (data['ma_fast'] > data['ma_slow']) & (data['ma_fast'].shift(1) <= data['ma_slow'].shift(1))
        
        # Sell signal: fast MA crosses below slow MA
        sell_signals = (data['ma_fast'] < data['ma_slow']) & (data['ma_fast'].shift(1) >= data['ma_slow'].shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals

class RSIStrategy(TradingStrategy):
    """RSI-based strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI", {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought
        })
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI"""
        data = data.copy()
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI"""
        data = self.calculate_indicators(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: RSI crosses above oversold level
        buy_signals = (data['rsi'] > self.oversold) & (data['rsi'].shift(1) <= self.oversold)
        
        # Sell signal: RSI crosses below overbought level
        sell_signals = (data['rsi'] < self.overbought) & (data['rsi'].shift(1) >= self.overbought)
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies
    """
    
    def __init__(self):
        self.config = get_config()
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage = 0.0005  # 0.05% slippage
        
    def run_backtest(self, strategy: TradingStrategy, data: pd.DataFrame,
                    initial_balance: float = 100000, stop_loss: float = None,
                    take_profit: float = None) -> BacktestResult:
        """Run a complete backtest"""
        
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Ensure data is sorted by time
        data = data.sort_index()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize tracking variables
        balance = initial_balance
        position = 0  # Current position size
        trades = []
        equity_curve = []
        current_trade = None
        
        # Performance tracking
        daily_returns = []
        max_balance = initial_balance
        max_drawdown = 0
        consecutive_losses = 0
        consecutive_wins = 0
        max_consecutive_losses = 0
        max_consecutive_wins = 0
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Calculate current portfolio value
            portfolio_value = balance + (position * current_price if position != 0 else 0)
            
            # Update max drawdown
            if portfolio_value > max_balance:
                max_balance = portfolio_value
            
            current_drawdown = (max_balance - portfolio_value) / max_balance
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Record equity curve
            equity_curve.append({
                'timestamp': timestamp,
                'balance': balance,
                'position_value': position * current_price if position != 0 else 0,
                'total_value': portfolio_value,
                'drawdown': current_drawdown
            })
            
            # Check stop loss / take profit
            if current_trade and position != 0:
                entry_price = current_trade.entry_price
                
                # Stop loss check
                if stop_loss:
                    if position > 0 and current_price <= entry_price * (1 - stop_loss):
                        signal = -1  # Force exit
                        current_trade.reason = 'stop_loss'
                    elif position < 0 and current_price >= entry_price * (1 + stop_loss):
                        signal = 1  # Force exit
                        current_trade.reason = 'stop_loss'
                
                # Take profit check
                if take_profit:
                    if position > 0 and current_price >= entry_price * (1 + take_profit):
                        signal = -1  # Force exit
                        current_trade.reason = 'take_profit'
                    elif position < 0 and current_price <= entry_price * (1 - take_profit):
                        signal = 1  # Force exit
                        current_trade.reason = 'take_profit'
            
            # Process signals
            if signal != 0:
                # Calculate transaction costs
                transaction_cost = abs(signal) * current_price * self.commission_rate
                slippage_cost = abs(signal) * current_price * self.slippage
                total_cost = transaction_cost + slippage_cost
                
                if signal > 0:  # Buy signal
                    if position <= 0:  # Enter long or close short
                        if position < 0:  # Close short position
                            pnl = position * (current_trade.entry_price - current_price)
                            balance += pnl - total_cost
                            
                            # Close current trade
                            current_trade.exit_time = timestamp
                            current_trade.exit_price = current_price
                            current_trade.pnl = pnl - total_cost
                            current_trade.status = 'closed'
                            trades.append(current_trade)
                            
                            # Update consecutive counters
                            if pnl > 0:
                                consecutive_wins += 1
                                consecutive_losses = 0
                                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                            else:
                                consecutive_losses += 1
                                consecutive_wins = 0
                                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        
                        # Enter long position
                        position_size = balance * 0.95 / current_price  # Use 95% of balance
                        position = position_size
                        balance -= position_size * current_price + total_cost
                        
                        current_trade = Trade(
                            entry_time=timestamp,
                            exit_time=None,
                            symbol='BACKTEST',
                            side='long',
                            entry_price=current_price,
                            exit_price=None,
                            quantity=position_size,
                            pnl=None,
                            status='open',
                            reason='signal'
                        )
                
                elif signal < 0:  # Sell signal
                    if position >= 0:  # Enter short or close long
                        if position > 0:  # Close long position
                            pnl = position * (current_price - current_trade.entry_price)
                            balance += position * current_price + pnl - total_cost
                            
                            # Close current trade
                            current_trade.exit_time = timestamp
                            current_trade.exit_price = current_price
                            current_trade.pnl = pnl - total_cost
                            current_trade.status = 'closed'
                            trades.append(current_trade)
                            
                            # Update consecutive counters
                            if pnl > 0:
                                consecutive_wins += 1
                                consecutive_losses = 0
                                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                            else:
                                consecutive_losses += 1
                                consecutive_wins = 0
                                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        
                        # Enter short position (simplified - assume we can short)
                        position_size = balance * 0.95 / current_price
                        position = -position_size
                        balance += position_size * current_price - total_cost
                        
                        current_trade = Trade(
                            entry_time=timestamp,
                            exit_time=None,
                            symbol='BACKTEST',
                            side='short',
                            entry_price=current_price,
                            exit_price=None,
                            quantity=position_size,
                            pnl=None,
                            status='open',
                            reason='signal'
                        )
        
        # Close any remaining position at the end
        if current_trade and position != 0:
            final_price = data['close'].iloc[-1]
            if position > 0:
                pnl = position * (final_price - current_trade.entry_price)
                balance += position * final_price + pnl
            else:
                pnl = position * (current_trade.entry_price - final_price)
                balance += pnl
            
            current_trade.exit_time = data.index[-1]
            current_trade.exit_price = final_price
            current_trade.pnl = pnl
            current_trade.status = 'closed'
            current_trade.reason = 'time_exit'
            trades.append(current_trade)
        
        # Calculate performance metrics
        final_balance = balance
        total_return = (final_balance - initial_balance) / initial_balance
        
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        avg_trade_return = np.mean([t.pnl for t in trades]) if trades else 0
        
        # Calculate Sharpe ratio
        if equity_curve:
            returns = []
            for i in range(1, len(equity_curve)):
                prev_value = equity_curve[i-1]['total_value']
                curr_value = equity_curve[i]['total_value']
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
            
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate Sortino ratio (downside deviation)
        if equity_curve:
            negative_returns = [r for r in returns if r < 0]
            if negative_returns and np.std(negative_returns) > 0:
                sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
        
        # Calculate Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum([t.pnl for t in trades if t.pnl > 0])
        gross_loss = abs(sum([t.pnl for t in trades if t.pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            parameters=strategy.parameters,
            trade_log=[{
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'symbol': t.symbol,
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'reason': t.reason
            } for t in trades],
            equity_curve=equity_curve
        )
        
        logger.info(f"Backtest completed for {strategy.name}: "
                   f"Return: {total_return:.2%}, Trades: {len(trades)}, "
                   f"Win Rate: {win_rate:.2%}, Sharpe: {sharpe_ratio:.2f}")
        
        return result
    
    def optimize_strategy(self, strategy_class, data: pd.DataFrame, 
                         parameter_ranges: Dict[str, List], 
                         optimization_metric: str = 'sharpe_ratio') -> List[BacktestResult]:
        """Optimize strategy parameters"""
        
        logger.info(f"Starting parameter optimization for {strategy_class.__name__}")
        
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        for i, param_combo in enumerate(param_combinations):
            try:
                # Create strategy with current parameters
                params = dict(zip(param_names, param_combo))
                strategy = strategy_class(**params)
                
                # Run backtest
                result = self.run_backtest(strategy, data)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(param_combinations)} optimizations")
                
            except Exception as e:
                logger.error(f"Error in optimization {i}: {e}")
                continue
        
        # Sort by optimization metric
        if results:
            results.sort(key=lambda x: getattr(x, optimization_metric), reverse=True)
            
            best_result = results[0]
            logger.info(f"Best parameters found: {best_result.parameters}")
            logger.info(f"Best {optimization_metric}: {getattr(best_result, optimization_metric):.4f}")
        
        return results
    
    def walk_forward_analysis(self, strategy: TradingStrategy, data: pd.DataFrame,
                            training_window: int = 252, testing_window: int = 63) -> List[BacktestResult]:
        """Perform walk-forward analysis"""
        
        logger.info("Starting walk-forward analysis")
        
        results = []
        start_idx = training_window
        
        while start_idx + testing_window <= len(data):
            try:
                # Training period
                train_data = data.iloc[start_idx-training_window:start_idx]
                
                # Testing period
                test_data = data.iloc[start_idx:start_idx+testing_window]
                
                # Run backtest on test period
                result = self.run_backtest(strategy, test_data)
                results.append(result)
                
                logger.info(f"Walk-forward period {len(results)}: "
                           f"Return: {result.total_return:.2%}")
                
                start_idx += testing_window
                
            except Exception as e:
                logger.error(f"Error in walk-forward analysis: {e}")
                start_idx += testing_window
                continue
        
        logger.info(f"Walk-forward analysis completed with {len(results)} periods")
        return results
    
    def monte_carlo_simulation(self, strategy: TradingStrategy, data: pd.DataFrame,
                              num_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation on trade sequences"""
        
        logger.info(f"Starting Monte Carlo simulation with {num_simulations} runs")
        
        # Run initial backtest to get trade sequence
        base_result = self.run_backtest(strategy, data)
        trade_returns = [t['pnl'] for t in base_result.trade_log]
        
        if not trade_returns:
            logger.warning("No trades found for Monte Carlo simulation")
            return {}
        
        simulation_results = []
        
        for i in range(num_simulations):
            # Randomly resample trade returns
            random_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate cumulative performance
            cumulative_balance = base_result.initial_balance
            equity_curve = [cumulative_balance]
            
            for trade_return in random_returns:
                cumulative_balance += trade_return
                equity_curve.append(cumulative_balance)
            
            final_return = (cumulative_balance - base_result.initial_balance) / base_result.initial_balance
            
            # Calculate max drawdown for this simulation
            max_balance = base_result.initial_balance
            max_dd = 0
            
            for balance in equity_curve:
                if balance > max_balance:
                    max_balance = balance
                current_dd = (max_balance - balance) / max_balance
                max_dd = max(max_dd, current_dd)
            
            simulation_results.append({
                'final_return': final_return,
                'max_drawdown': max_dd,
                'final_balance': cumulative_balance
            })
        
        # Calculate statistics
        returns = [sim['final_return'] for sim in simulation_results]
        drawdowns = [sim['max_drawdown'] for sim in simulation_results]
        
        monte_carlo_stats = {
            'num_simulations': num_simulations,
            'return_stats': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95)
            },
            'drawdown_stats': {
                'mean': np.mean(drawdowns),
                'std': np.std(drawdowns),
                'min': np.min(drawdowns),
                'max': np.max(drawdowns),
                'percentile_95': np.percentile(drawdowns, 95)
            },
            'probability_positive': len([r for r in returns if r > 0]) / len(returns)
        }
        
        logger.info(f"Monte Carlo simulation completed. "
                   f"Mean return: {monte_carlo_stats['return_stats']['mean']:.2%}, "
                   f"95% VaR: {monte_carlo_stats['return_stats']['percentile_5']:.2%}")
        
        return monte_carlo_stats

# Global backtesting engine instance
backtest_engine = None

def get_backtest_engine() -> BacktestEngine:
    """Get the global backtest engine instance"""
    global backtest_engine
    if backtest_engine is None:
        backtest_engine = BacktestEngine()
    return backtest_engine