
import numpy as np
import datetime
from typing import Dict, List, Optional, Any
from config import config


class ExternalRiskManager:
    """Enhanced risk management system with portfolio risk, position sizing, and stop-loss mechanisms."""
    
    def __init__(self, volatility_threshold=0.8, news_impact_weight=0.6):
        """
        Initialize the risk manager with enhanced features.
        
        Args:
            volatility_threshold: Maximum allowed volatility level
            news_impact_weight: Weight for news impact in risk calculation
        """
        self.volatility_threshold = volatility_threshold
        self.news_impact_weight = news_impact_weight
        self.current_volatility = 0.0
        self.market_event_risk = 0.0
        self.last_decision = None
        
        # Enhanced features
        self.portfolio = {}  # {symbol: {'position': float, 'entry_price': float, 'stop_loss': float}}
        self.account_balance = config.initial_capital
        self.max_risk_per_trade = config.risk_percentage / 100
        self.max_positions = config.max_positions
        self.total_exposure = 0.0
        
        # Risk metrics history
        self.risk_history = []
        self.volatility_history = []
        
    def update_volatility(self, prices):
        """Update volatility calculation with enhanced metrics."""
        returns = np.diff(prices) / prices[:-1]
        self.current_volatility = np.std(returns[-20:])
        
        # Track volatility history
        self.volatility_history.append({
            'timestamp': datetime.datetime.now(),
            'volatility': self.current_volatility
        })
        
        # Keep only last 100 records
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)

    def update_news_impact(self, impact_score):
        """Update news impact assessment."""
        self.market_event_risk = impact_score

    def calculate_portfolio_risk(self) -> float:
        """
        Calculate overall portfolio risk.
        
        Returns:
            Portfolio risk score (0-1)
        """
        if not self.portfolio:
            return 0.0
        
        # Calculate position correlation risk
        total_exposure = sum(abs(pos['position']) for pos in self.portfolio.values())
        exposure_ratio = total_exposure / self.account_balance if self.account_balance > 0 else 0
        
        # Calculate concentration risk
        concentration_risk = 0.0
        if len(self.portfolio) > 0:
            max_position = max(abs(pos['position']) for pos in self.portfolio.values())
            concentration_risk = max_position / self.account_balance if self.account_balance > 0 else 0
        
        # Combine risks
        portfolio_risk = min(exposure_ratio * 0.6 + concentration_risk * 0.4, 1.0)
        return portfolio_risk

    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float) -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price level
            
        Returns:
            Recommended position size
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            return 0.0
        
        # Maximum risk amount for this trade
        max_risk_amount = self.account_balance * self.max_risk_per_trade
        
        # Calculate position size
        position_size = max_risk_amount / risk_per_unit
        
        # Apply portfolio constraints
        portfolio_risk = self.calculate_portfolio_risk()
        if portfolio_risk > 0.7:  # High portfolio risk
            position_size *= 0.5  # Reduce position size
        
        # Check maximum positions limit
        if len(self.portfolio) >= self.max_positions:
            return 0.0
        
        return position_size

    def set_stop_loss(self, symbol: str, entry_price: float, 
                     position_type: str = 'long') -> float:
        """
        Calculate stop loss level based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            
        Returns:
            Stop loss price level
        """
        stop_loss_percentage = config.stop_loss_percentage / 100
        
        if position_type.lower() == 'long':
            stop_loss = entry_price * (1 - stop_loss_percentage)
        else:  # short position
            stop_loss = entry_price * (1 + stop_loss_percentage)
        
        return stop_loss

    def set_take_profit(self, symbol: str, entry_price: float, 
                       position_type: str = 'long') -> float:
        """
        Calculate take profit level based on risk-reward ratio.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'long' or 'short'
            
        Returns:
            Take profit price level
        """
        take_profit_percentage = config.take_profit_percentage / 100
        
        if position_type.lower() == 'long':
            take_profit = entry_price * (1 + take_profit_percentage)
        else:  # short position
            take_profit = entry_price * (1 - take_profit_percentage)
        
        return take_profit

    def add_position(self, symbol: str, position_size: float, entry_price: float,
                    position_type: str = 'long') -> bool:
        """
        Add a new position to the portfolio.
        
        Args:
            symbol: Trading symbol
            position_size: Size of the position
            entry_price: Entry price
            position_type: 'long' or 'short'
            
        Returns:
            True if position added successfully, False otherwise
        """
        # Check if we can add more positions
        if len(self.portfolio) >= self.max_positions:
            return False
        
        # Calculate stop loss and take profit
        stop_loss = self.set_stop_loss(symbol, entry_price, position_type)
        take_profit = self.set_take_profit(symbol, entry_price, position_type)
        
        # Add position to portfolio
        self.portfolio[symbol] = {
            'position': position_size if position_type == 'long' else -position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_type': position_type,
            'timestamp': datetime.datetime.now()
        }
        
        # Update account balance (assuming margin trading)
        self.total_exposure += abs(position_size * entry_price)
        
        return True

    def close_position(self, symbol: str, exit_price: float) -> Dict[str, Any]:
        """
        Close a position and calculate P&L.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            
        Returns:
            Dictionary with position closure details
        """
        if symbol not in self.portfolio:
            return {'success': False, 'message': 'Position not found'}
        
        position = self.portfolio[symbol]
        position_size = abs(position['position'])
        entry_price = position['entry_price']
        
        # Calculate P&L
        if position['position'] > 0:  # Long position
            pnl = (exit_price - entry_price) * position_size
        else:  # Short position
            pnl = (entry_price - exit_price) * position_size
        
        # Update account balance
        self.account_balance += pnl
        self.total_exposure -= position_size * entry_price
        
        # Remove position from portfolio
        closed_position = self.portfolio.pop(symbol)
        
        return {
            'success': True,
            'symbol': symbol,
            'pnl': pnl,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'hold_time': datetime.datetime.now() - closed_position['timestamp']
        }

    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check if any positions need to be closed due to stop losses.
        
        Args:
            current_prices: Dictionary of current prices by symbol
            
        Returns:
            List of symbols that hit stop loss
        """
        stop_loss_hits = []
        
        for symbol, position in self.portfolio.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Check stop loss condition
                if position['position'] > 0:  # Long position
                    if current_price <= position['stop_loss']:
                        stop_loss_hits.append(symbol)
                else:  # Short position
                    if current_price >= position['stop_loss']:
                        stop_loss_hits.append(symbol)
        
        return stop_loss_hits

    def multi_timeframe_analysis(self, timeframes_data: Dict[str, Any]) -> float:
        """
        Analyze risk across multiple timeframes.
        
        Args:
            timeframes_data: Dictionary with data for different timeframes
            
        Returns:
            Combined risk score across timeframes
        """
        timeframe_risks = []
        
        for timeframe, data in timeframes_data.items():
            if 'volatility' in data:
                # Weight shorter timeframes more heavily for immediate risk
                weight = 0.5 if timeframe in ['1m', '5m'] else 0.3 if timeframe in ['15m', '1h'] else 0.2
                timeframe_risk = data['volatility'] * weight
                timeframe_risks.append(timeframe_risk)
        
        return sum(timeframe_risks) if timeframe_risks else 0.0

    def assess_risk(self):
        """Enhanced risk assessment with portfolio and multi-factor analysis."""
        # Base risk from volatility and news
        base_risk = (
            self.current_volatility +
            self.market_event_risk * self.news_impact_weight
        )
        
        # Add portfolio risk
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Combine all risk factors
        total_risk = min(base_risk * 0.6 + portfolio_risk * 0.4, 1.0)
        
        # Record risk history
        self.risk_history.append({
            'timestamp': datetime.datetime.now(),
            'total_risk': total_risk,
            'base_risk': base_risk,
            'portfolio_risk': portfolio_risk
        })
        
        return total_risk

    def should_halt_trading(self):
        """Enhanced trading halt decision."""
        total_risk = self.assess_risk()
        
        # Additional halt conditions
        portfolio_risk = self.calculate_portfolio_risk()
        account_drawdown = (config.initial_capital - self.account_balance) / config.initial_capital
        
        halt_conditions = [
            total_risk >= self.volatility_threshold,
            portfolio_risk > 0.8,
            account_drawdown > 0.2,  # 20% drawdown
            len(self.portfolio) >= self.max_positions
        ]
        
        return any(halt_conditions)

    def generate_signal(self):
        """Enhanced signal generation with detailed risk assessment."""
        if self.should_halt_trading():
            risk_details = {
                'total_risk': self.assess_risk(),
                'portfolio_risk': self.calculate_portfolio_risk(),
                'account_balance': self.account_balance,
                'positions_count': len(self.portfolio),
                'total_exposure': self.total_exposure
            }
            
            return {
                "action": "HOLD", 
                "reason": "High Risk", 
                "timestamp": str(datetime.datetime.now()),
                "risk_details": risk_details
            }
        
        return {
            "action": "PROCEED", 
            "risk_score": self.assess_risk(),
            "portfolio_status": {
                'positions': len(self.portfolio),
                'account_balance': self.account_balance,
                'total_exposure': self.total_exposure
            }
        }

    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Returns:
            Dictionary with detailed risk metrics
        """
        return {
            'current_risk': self.assess_risk(),
            'portfolio_risk': self.calculate_portfolio_risk(),
            'account_balance': self.account_balance,
            'total_exposure': self.total_exposure,
            'positions_count': len(self.portfolio),
            'max_positions': self.max_positions,
            'volatility': self.current_volatility,
            'volatility_threshold': self.volatility_threshold,
            'positions': {symbol: {
                'size': pos['position'],
                'entry_price': pos['entry_price'],
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit']
            } for symbol, pos in self.portfolio.items()},
            'risk_history_length': len(self.risk_history)
        }
