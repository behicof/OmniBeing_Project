"""
Backtesting System for the OmniBeing Trading System.
Provides historical strategy testing, performance metrics, and optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass
from config import config
from data_manager import DataManager
from external_risk_manager import ExternalRiskManager


@dataclass
class Trade:
    """Data class for trade records."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    position_type: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    pnl: Optional[float] = None
    status: str = 'open'  # 'open', 'closed', 'stopped'


class BacktestingEngine:
    """
    Backtesting engine for strategy testing and optimization.
    """
    
    def __init__(self, initial_capital: float = None):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital or config.initial_capital
        self.current_capital = self.initial_capital
        self.data_manager = DataManager()
        
        # Backtesting state
        self.trades: List[Trade] = []
        self.portfolio_value_history = []
        self.current_time = None
        self.historical_data = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.daily_returns = []
        self.equity_curve = []
        
    def load_historical_data(self, symbol: str, start_date: datetime = None, 
                           end_date: datetime = None, timeframe: str = '1h') -> pd.DataFrame:
        """
        Load historical data for backtesting.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backtesting
            end_date: End date for backtesting
            timeframe: Data timeframe
            
        Returns:
            DataFrame with historical data
        """
        # For demonstration, we'll use the data manager's fetch method
        # In a real implementation, this would load actual historical data
        data = self.data_manager.fetch_historical_data(symbol, timeframe, 2000)
        
        # Filter by date range if provided
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Add technical indicators
        data = self.data_manager.calculate_technical_indicators(data)
        
        self.historical_data = data
        return data
    
    def create_strategy_signals(self, strategy_func: Callable, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using a strategy function.
        
        Args:
            strategy_func: Function that takes market data and returns signals
            data: Historical market data
            
        Returns:
            DataFrame with signals added
        """
        signals = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Create market data dictionary for strategy function
            market_data = {
                'price': row['close'],
                'volume': row['volume'],
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            
            # Add technical indicators if available
            for col in ['sma_20', 'sma_50', 'rsi', 'macd', 'volatility']:
                if col in row and not pd.isna(row[col]):
                    market_data[col] = row[col]
            
            # Get signal from strategy
            try:
                signal = strategy_func(market_data, i)
                signals.append(signal)
            except Exception as e:
                signals.append('hold')  # Default to hold on error
        
        data['signal'] = signals
        return data
    
    def simple_ma_crossover_strategy(self, market_data: Dict[str, Any], index: int) -> str:
        """
        Simple moving average crossover strategy for demonstration.
        
        Args:
            market_data: Current market data
            index: Current index in the data
            
        Returns:
            Trading signal ('buy', 'sell', 'hold')
        """
        if 'sma_20' not in market_data or 'sma_50' not in market_data:
            return 'hold'
        
        sma_20 = market_data['sma_20']
        sma_50 = market_data['sma_50']
        
        if pd.isna(sma_20) or pd.isna(sma_50):
            return 'hold'
        
        # Buy when short MA crosses above long MA
        if sma_20 > sma_50:
            return 'buy'
        # Sell when short MA crosses below long MA
        elif sma_20 < sma_50:
            return 'sell'
        else:
            return 'hold'
    
    def rsi_strategy(self, market_data: Dict[str, Any], index: int) -> str:
        """
        RSI-based trading strategy.
        
        Args:
            market_data: Current market data
            index: Current index in the data
            
        Returns:
            Trading signal ('buy', 'sell', 'hold')
        """
        if 'rsi' not in market_data:
            return 'hold'
        
        rsi = market_data['rsi']
        
        if pd.isna(rsi):
            return 'hold'
        
        # Buy when RSI is oversold
        if rsi < 30:
            return 'buy'
        # Sell when RSI is overbought
        elif rsi > 70:
            return 'sell'
        else:
            return 'hold'
    
    def run_backtest(self, symbol: str, strategy_func: Callable = None, 
                    start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        Run a complete backtest.
        
        Args:
            symbol: Trading symbol
            strategy_func: Strategy function to test
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        # Use default strategy if none provided
        if strategy_func is None:
            strategy_func = self.simple_ma_crossover_strategy
        
        # Reset state
        self.trades = []
        self.current_capital = self.initial_capital
        self.portfolio_value_history = []
        
        # Load historical data
        data = self.load_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            return {'error': 'No historical data available'}
        
        # Generate signals
        data = self.create_strategy_signals(strategy_func, data)
        
        # Initialize risk manager for backtesting
        risk_manager = ExternalRiskManager()
        risk_manager.account_balance = self.initial_capital
        
        # Track portfolio value
        portfolio_values = []
        
        # Execute backtest
        for i, (timestamp, row) in enumerate(data.iterrows()):
            self.current_time = timestamp
            current_price = row['close']
            signal = row['signal']
            
            # Update risk manager with current price data
            if i > 20:  # Need some history for volatility calculation
                recent_prices = data['close'].iloc[max(0, i-20):i+1].values
                risk_manager.update_volatility(recent_prices)
            
            # Check for position exits (stop loss, take profit)
            self._check_position_exits(risk_manager, current_price, timestamp)
            
            # Process new signals
            if signal in ['buy', 'sell'] and len(risk_manager.portfolio) < config.max_positions:
                self._execute_backtest_trade(risk_manager, symbol, signal, current_price, timestamp)
            
            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value(risk_manager, current_price)
            portfolio_values.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'price': current_price
            })
        
        # Close any remaining positions
        final_price = data['close'].iloc[-1]
        self._close_all_positions(risk_manager, final_price, data.index[-1])
        
        # Calculate performance metrics
        self.portfolio_value_history = portfolio_values
        self.performance_metrics = self._calculate_performance_metrics()
        
        return {
            'symbol': symbol,
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'initial_capital': self.initial_capital,
            'final_capital': risk_manager.account_balance,
            'total_return': (risk_manager.account_balance - self.initial_capital) / self.initial_capital * 100,
            'total_trades': len(self.trades),
            'performance_metrics': self.performance_metrics,
            'trades': [self._trade_to_dict(trade) for trade in self.trades],
            'portfolio_history': portfolio_values
        }
    
    def _execute_backtest_trade(self, risk_manager: ExternalRiskManager, symbol: str, 
                               signal: str, price: float, timestamp: datetime):
        """Execute a trade in the backtest."""
        position_type = 'long' if signal == 'buy' else 'short'
        
        # Calculate stop loss and take profit
        stop_loss = risk_manager.set_stop_loss(symbol, price, position_type)
        take_profit = risk_manager.set_take_profit(symbol, price, position_type)
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(symbol, price, stop_loss)
        
        if position_size > 0:
            # Add position to risk manager
            success = risk_manager.add_position(symbol, position_size, price, position_type)
            
            if success:
                # Create trade record
                trade = Trade(
                    symbol=symbol,
                    entry_time=timestamp,
                    exit_time=None,
                    entry_price=price,
                    exit_price=None,
                    position_size=position_size,
                    position_type=position_type,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                self.trades.append(trade)
    
    def _check_position_exits(self, risk_manager: ExternalRiskManager, 
                             current_price: float, timestamp: datetime):
        """Check and execute position exits."""
        positions_to_close = []
        
        for symbol, position in risk_manager.portfolio.items():
            should_close = False
            exit_reason = ""
            
            # Check stop loss
            if position['position'] > 0:  # Long position
                if current_price <= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price >= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            else:  # Short position
                if current_price >= position['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price <= position['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                positions_to_close.append((symbol, exit_reason))
        
        # Close positions
        for symbol, exit_reason in positions_to_close:
            result = risk_manager.close_position(symbol, current_price)
            if result['success']:
                # Update trade record
                for trade in reversed(self.trades):
                    if (trade.symbol == symbol and trade.exit_time is None and 
                        trade.status == 'open'):
                        trade.exit_time = timestamp
                        trade.exit_price = current_price
                        trade.pnl = result['pnl']
                        trade.status = exit_reason
                        break
    
    def _close_all_positions(self, risk_manager: ExternalRiskManager, 
                           final_price: float, final_timestamp: datetime):
        """Close all remaining positions at the end of backtest."""
        for symbol in list(risk_manager.portfolio.keys()):
            result = risk_manager.close_position(symbol, final_price)
            if result['success']:
                # Update trade record
                for trade in reversed(self.trades):
                    if (trade.symbol == symbol and trade.exit_time is None and 
                        trade.status == 'open'):
                        trade.exit_time = final_timestamp
                        trade.exit_price = final_price
                        trade.pnl = result['pnl']
                        trade.status = 'closed'
                        break
    
    def _calculate_portfolio_value(self, risk_manager: ExternalRiskManager, 
                                  current_price: float) -> float:
        """Calculate current portfolio value."""
        return risk_manager.account_balance
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_value_history or not self.trades:
            return {}
        
        # Extract portfolio values
        values = [pv['portfolio_value'] for pv in self.portfolio_value_history]
        
        # Calculate returns
        returns = np.diff(values) / values[:-1]
        
        # Calculate metrics
        completed_trades = [t for t in self.trades if t.pnl is not None]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in completed_trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate maximum drawdown
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return {
            'total_return': (values[-1] - values[0]) / values[0] * 100,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0,
            'total_pnl': total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'maximum_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': values[-1] if values else self.initial_capital
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert Trade object to dictionary."""
        return {
            'symbol': trade.symbol,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_size': trade.position_size,
            'position_type': trade.position_type,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'pnl': trade.pnl,
            'status': trade.status
        }
    
    def plot_results(self, save_path: str = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.portfolio_value_history:
            print("No backtest results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Portfolio value over time
        timestamps = [pv['timestamp'] for pv in self.portfolio_value_history]
        values = [pv['portfolio_value'] for pv in self.portfolio_value_history]
        prices = [pv['price'] for pv in self.portfolio_value_history]
        
        ax1.plot(timestamps, values, label='Portfolio Value', linewidth=2)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Price chart with trade markers
        ax2.plot(timestamps, prices, label='Price', alpha=0.7)
        
        # Mark buy and sell signals
        for trade in self.trades:
            if trade.position_type == 'long':
                ax2.scatter(trade.entry_time, trade.entry_price, color='green', marker='^', s=100, label='Buy' if trade == self.trades[0] else "")
                if trade.exit_time:
                    ax2.scatter(trade.exit_time, trade.exit_price, color='red', marker='v', s=100, label='Sell' if trade == self.trades[0] else "")
        
        ax2.set_title('Price Chart with Trades')
        ax2.set_ylabel('Price')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def optimize_strategy(self, symbol: str, strategy_func: Callable, 
                         parameter_ranges: Dict[str, List], 
                         optimization_metric: str = 'total_return') -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            symbol: Trading symbol
            strategy_func: Strategy function to optimize
            parameter_ranges: Dictionary of parameter ranges to test
            optimization_metric: Metric to optimize for
            
        Returns:
            Dictionary with optimization results
        """
        best_params = None
        best_score = float('-inf')
        all_results = []
        
        # Generate parameter combinations
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        for param_combination in itertools.product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            # Create strategy function with current parameters
            def parameterized_strategy(market_data, index):
                return strategy_func(market_data, index, **params)
            
            # Run backtest
            try:
                result = self.run_backtest(symbol, parameterized_strategy)
                
                if 'performance_metrics' in result:
                    score = result['performance_metrics'].get(optimization_metric, float('-inf'))
                    
                    all_results.append({
                        'parameters': params,
                        'score': score,
                        'metrics': result['performance_metrics']
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
            except Exception as e:
                print(f"Error testing parameters {params}: {e}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_metric': optimization_metric,
            'all_results': all_results
        }