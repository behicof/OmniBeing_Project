"""
Main Trading System Controller for the OmniBeing Trading System.
Integrates all modules and provides unified API for trading operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import threading
import time

# Import existing modules
from config import config
from data_manager import DataManager
from external_risk_manager import ExternalRiskManager

# Import existing prediction systems
try:
    from advanced_predictive_system import AdvancedPredictiveSystem
except ImportError:
    AdvancedPredictiveSystem = None

try:
    from final_real_time_optimization_predictive_system import FinalRealTimeOptimizationPredictiveSystem
except ImportError:
    FinalRealTimeOptimizationPredictiveSystem = None


class MainTradingSystem:
    """
    Main controller that orchestrates all components of the trading system.
    Provides clean API for external usage and manages system lifecycle.
    """
    
    def __init__(self):
        """Initialize the main trading system with all components."""
        self.logger = self._setup_logging()
        
        # Core components
        self.config = config
        self.data_manager = DataManager()
        self.risk_manager = ExternalRiskManager(
            volatility_threshold=config.volatility_threshold,
            news_impact_weight=0.6
        )
        
        # Prediction systems
        self.prediction_systems = {}
        self._initialize_prediction_systems()
        
        # System state
        self.is_running = False
        self.is_trading_enabled = True
        self.last_prediction = None
        self.last_risk_assessment = None
        
        # Performance tracking
        self.trades_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0
        }
        
        # Threading for real-time operations
        self.main_thread = None
        self.stop_event = threading.Event()
        
        self.logger.info("MainTradingSystem initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('TradingSystem')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_prediction_systems(self):
        """Initialize available prediction systems."""
        try:
            if AdvancedPredictiveSystem:
                self.prediction_systems['advanced'] = AdvancedPredictiveSystem()
                self.logger.info("Initialized AdvancedPredictiveSystem")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AdvancedPredictiveSystem: {e}")
        
        try:
            if FinalRealTimeOptimizationPredictiveSystem:
                self.prediction_systems['realtime'] = FinalRealTimeOptimizationPredictiveSystem()
                self.logger.info("Initialized FinalRealTimeOptimizationPredictiveSystem")
        except Exception as e:
            self.logger.warning(f"Failed to initialize FinalRealTimeOptimizationPredictiveSystem: {e}")
    
    def get_market_data(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get current market data for analysis.
        
        Args:
            symbol: Trading symbol (defaults to configured instrument)
            
        Returns:
            Dictionary with current market data
        """
        if symbol is None:
            symbol = self.config.trading_instrument
        
        try:
            market_data = self.data_manager.get_market_data_for_prediction(symbol)
            self.logger.debug(f"Retrieved market data for {symbol}")
            return market_data
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
    
    def make_prediction(self, symbol: str = None) -> Dict[str, Any]:
        """
        Generate trading predictions using available prediction systems.
        
        Args:
            symbol: Trading symbol (defaults to configured instrument)
            
        Returns:
            Dictionary with predictions from all systems
        """
        if symbol is None:
            symbol = self.config.trading_instrument
        
        try:
            market_data = self.get_market_data(symbol)
            if not market_data:
                return {'error': 'No market data available'}
            
            predictions = {}
            
            # Get predictions from all available systems
            for system_name, system in self.prediction_systems.items():
                try:
                    # Process market data for the system
                    if hasattr(system, 'process_market_data'):
                        system.process_market_data(market_data)
                    
                    # Train if needed (for systems that support incremental training)
                    if hasattr(system, 'train_models') and len(system.market_data) > 10:
                        system.train_models()
                    
                    # Make prediction
                    if hasattr(system, 'make_predictions'):
                        prediction = system.make_predictions(market_data)
                        predictions[system_name] = prediction
                    
                except Exception as e:
                    self.logger.error(f"Error with prediction system {system_name}: {e}")
                    predictions[system_name] = 'error'
            
            # Combine predictions with ensemble method
            combined_prediction = self._combine_predictions(predictions)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'individual_predictions': predictions,
                'combined_prediction': combined_prediction,
                'market_data': market_data
            }
            
            self.last_prediction = result
            self.logger.info(f"Generated prediction for {symbol}: {combined_prediction}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            return {'error': str(e)}
    
    def _combine_predictions(self, predictions: Dict[str, Any]) -> str:
        """
        Combine predictions from multiple systems using ensemble method.
        
        Args:
            predictions: Dictionary of predictions from different systems
            
        Returns:
            Combined prediction
        """
        if not predictions:
            return 'hold'
        
        # Simple majority voting
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        
        for system_name, prediction in predictions.items():
            if prediction == 'buy' or prediction == 1:
                buy_votes += 1
            elif prediction == 'sell' or prediction == 0:
                sell_votes += 1
            else:
                hold_votes += 1
        
        # Return prediction with most votes
        if buy_votes > sell_votes and buy_votes > hold_votes:
            return 'buy'
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            return 'sell'
        else:
            return 'hold'
    
    def assess_risk(self, symbol: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment.
        
        Args:
            symbol: Trading symbol (defaults to configured instrument)
            
        Returns:
            Dictionary with risk assessment results
        """
        if symbol is None:
            symbol = self.config.trading_instrument
        
        try:
            # Get current market data
            market_data = self.get_market_data(symbol)
            
            # Update risk manager with current data
            if 'price' in market_data:
                # Generate price series for volatility calculation
                historical_data = self.data_manager.historical_data.get(symbol)
                if historical_data is not None and len(historical_data) > 0:
                    prices = historical_data['close'].values[-100:]  # Last 100 prices
                    self.risk_manager.update_volatility(prices)
            
            # Get risk signal
            risk_signal = self.risk_manager.generate_signal()
            
            # Get comprehensive risk report
            risk_report = self.risk_manager.get_risk_report()
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'risk_signal': risk_signal,
                'risk_report': risk_report,
                'market_data': market_data
            }
            
            self.last_risk_assessment = result
            self.logger.debug(f"Risk assessment for {symbol}: {risk_signal['action']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error assessing risk for {symbol}: {e}")
            return {'error': str(e)}
    
    def execute_trade(self, signal: Dict[str, Any], symbol: str = None) -> Dict[str, Any]:
        """
        Execute a trade based on prediction and risk assessment.
        
        Args:
            signal: Trading signal dictionary
            symbol: Trading symbol (defaults to configured instrument)
            
        Returns:
            Dictionary with trade execution results
        """
        if symbol is None:
            symbol = self.config.trading_instrument
        
        try:
            # Check if trading is enabled
            if not self.is_trading_enabled:
                return {'status': 'disabled', 'message': 'Trading is disabled'}
            
            # Get current risk assessment
            risk_assessment = self.assess_risk(symbol)
            
            # Check risk conditions
            if risk_assessment.get('risk_signal', {}).get('action') == 'HOLD':
                return {
                    'status': 'blocked',
                    'message': 'Trade blocked by risk management',
                    'risk_reason': risk_assessment.get('risk_signal', {}).get('reason', 'Unknown')
                }
            
            # Get current market data
            market_data = self.get_market_data(symbol)
            current_price = market_data.get('price', 0)
            
            if current_price <= 0:
                return {'status': 'error', 'message': 'Invalid price data'}
            
            # Determine trade details
            action = signal.get('combined_prediction', 'hold')
            
            if action == 'buy':
                position_type = 'long'
                stop_loss_price = self.risk_manager.set_stop_loss(symbol, current_price, 'long')
                take_profit_price = self.risk_manager.set_take_profit(symbol, current_price, 'long')
            elif action == 'sell':
                position_type = 'short'
                stop_loss_price = self.risk_manager.set_stop_loss(symbol, current_price, 'short')
                take_profit_price = self.risk_manager.set_take_profit(symbol, current_price, 'short')
            else:
                return {'status': 'no_action', 'message': 'No trading signal generated'}
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, current_price, stop_loss_price
            )
            
            if position_size <= 0:
                return {'status': 'blocked', 'message': 'Position size calculation resulted in zero/negative size'}
            
            # Execute the trade (simulation mode for now)
            trade_result = self._simulate_trade_execution(
                symbol, action, position_size, current_price, stop_loss_price, take_profit_price
            )
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'position_size': position_size,
                'entry_price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'signal_data': signal,
                'risk_data': risk_assessment,
                'execution_result': trade_result
            }
            
            self.trades_history.append(trade_record)
            self.performance_metrics['total_trades'] += 1
            
            self.logger.info(f"Executed {action} trade for {symbol} at {current_price}")
            
            return {
                'status': 'executed',
                'trade_details': trade_record,
                'execution_result': trade_result
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _simulate_trade_execution(self, symbol: str, action: str, position_size: float,
                                 entry_price: float, stop_loss: float, take_profit: float) -> Dict[str, Any]:
        """
        Simulate trade execution (replace with real broker integration).
        
        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            position_size: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Dictionary with execution results
        """
        # Add position to risk manager
        success = self.risk_manager.add_position(symbol, position_size, entry_price, 
                                                'long' if action == 'buy' else 'short')
        
        if success:
            return {
                'status': 'filled',
                'fill_price': entry_price,
                'fill_size': position_size,
                'order_id': f"SIM_{int(datetime.now().timestamp())}"
            }
        else:
            return {
                'status': 'rejected',
                'reason': 'Risk manager rejected position'
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and performance metrics.
        
        Returns:
            Dictionary with system status
        """
        return {
            'is_running': self.is_running,
            'is_trading_enabled': self.is_trading_enabled,
            'prediction_systems': list(self.prediction_systems.keys()),
            'last_prediction': self.last_prediction,
            'last_risk_assessment': self.last_risk_assessment,
            'performance_metrics': self.performance_metrics,
            'active_positions': len(self.risk_manager.portfolio),
            'account_balance': self.risk_manager.account_balance,
            'total_exposure': self.risk_manager.total_exposure
        }
    
    def start_real_time_trading(self, update_interval: int = 60):
        """
        Start real-time trading loop.
        
        Args:
            update_interval: Update interval in seconds
        """
        if self.is_running:
            self.logger.warning("Real-time trading is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        def trading_loop():
            self.logger.info("Started real-time trading loop")
            
            while not self.stop_event.is_set():
                try:
                    # Make prediction
                    prediction = self.make_prediction()
                    
                    # Execute trade if conditions are met
                    if prediction and 'combined_prediction' in prediction:
                        if prediction['combined_prediction'] in ['buy', 'sell']:
                            trade_result = self.execute_trade(prediction)
                            self.logger.info(f"Trade execution result: {trade_result.get('status', 'unknown')}")
                    
                    # Check for stop losses
                    current_prices = {self.config.trading_instrument: 
                                    self.get_market_data().get('price', 0)}
                    stop_loss_hits = self.risk_manager.check_stop_losses(current_prices)
                    
                    # Close positions that hit stop loss
                    for symbol in stop_loss_hits:
                        if symbol in current_prices:
                            close_result = self.risk_manager.close_position(symbol, current_prices[symbol])
                            self.logger.info(f"Closed position {symbol} due to stop loss: {close_result}")
                    
                    # Wait for next update
                    self.stop_event.wait(update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    self.stop_event.wait(update_interval)
            
            self.logger.info("Real-time trading loop stopped")
        
        self.main_thread = threading.Thread(target=trading_loop, daemon=True)
        self.main_thread.start()
    
    def stop_real_time_trading(self):
        """Stop real-time trading loop."""
        if not self.is_running:
            self.logger.warning("Real-time trading is not running")
            return
        
        self.stop_event.set()
        self.is_running = False
        
        if self.main_thread:
            self.main_thread.join(timeout=5)
        
        self.logger.info("Real-time trading stopped")
    
    def enable_trading(self):
        """Enable trading operations."""
        self.is_trading_enabled = True
        self.logger.info("Trading enabled")
    
    def disable_trading(self):
        """Disable trading operations."""
        self.is_trading_enabled = False
        self.logger.info("Trading disabled")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades_history:
            return {'message': 'No trades executed yet'}
        
        # Calculate performance metrics
        total_pnl = sum(trade.get('execution_result', {}).get('pnl', 0) 
                       for trade in self.trades_history)
        
        winning_trades = sum(1 for trade in self.trades_history 
                           if trade.get('execution_result', {}).get('pnl', 0) > 0)
        
        win_rate = (winning_trades / len(self.trades_history)) * 100 if self.trades_history else 0
        
        return {
            'total_trades': len(self.trades_history),
            'winning_trades': winning_trades,
            'losing_trades': len(self.trades_history) - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_pnl_per_trade': total_pnl / len(self.trades_history) if self.trades_history else 0,
            'current_balance': self.risk_manager.account_balance,
            'active_positions': len(self.risk_manager.portfolio),
            'risk_report': self.risk_manager.get_risk_report()
        }
    
    def shutdown(self):
        """Shutdown the trading system gracefully."""
        self.logger.info("Shutting down trading system...")
        
        # Stop real-time trading
        self.stop_real_time_trading()
        
        # Close all open positions (in simulation mode)
        if self.risk_manager.portfolio:
            self.logger.info(f"Closing {len(self.risk_manager.portfolio)} open positions")
            # In a real system, this would close positions through broker API
        
        self.logger.info("Trading system shutdown complete")


# Create global trading system instance
trading_system = MainTradingSystem()