"""
Enhanced Risk Manager
Extends the existing ExternalRiskManager with advanced portfolio risk management
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

from external_risk_manager import ExternalRiskManager
from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    total_exposure: float
    position_risk: float
    correlation_risk: float
    volatility_risk: float
    news_risk: float
    drawdown_risk: float
    overall_risk_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class PositionRisk:
    """Individual position risk data"""
    symbol: str
    position_size: float
    risk_amount: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    max_loss_percent: float
    current_risk_score: float

class EnhancedRiskManager(ExternalRiskManager):
    """
    Advanced risk management system that extends the base ExternalRiskManager
    """
    
    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        self.config = get_config()
        
        # Enhanced risk parameters
        self.max_portfolio_risk = self.config.MAX_DAILY_LOSS
        self.max_position_size = self.config.MAX_POSITION_SIZE
        self.position_size_limit = self.config.DEFAULT_RISK_LEVEL
        
        # Risk tracking
        self.portfolio_value = 100000.0  # Starting portfolio value
        self.positions = {}
        self.risk_history = []
        self.correlation_matrix = {}
        self.drawdown_history = []
        
        # Dynamic risk adjustments
        self.risk_multiplier = 1.0
        self.emergency_stop = False
        
        logger.info("Enhanced Risk Manager initialized")
    
    def update_portfolio_value(self, new_value: float):
        """Update current portfolio value"""
        if self.portfolio_value > 0:
            # Calculate drawdown
            drawdown = (new_value - self.portfolio_value) / self.portfolio_value
            self.drawdown_history.append({
                'timestamp': datetime.now(),
                'value': new_value,
                'drawdown': drawdown
            })
            
            # Keep only last 1000 records
            if len(self.drawdown_history) > 1000:
                self.drawdown_history = self.drawdown_history[-1000:]
        
        self.portfolio_value = new_value
    
    def add_position(self, symbol: str, size: float, entry_price: float,
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """Add or update a position in the portfolio"""
        risk_amount = abs(size * entry_price)
        
        if stop_loss:
            max_loss = abs(entry_price - stop_loss) * abs(size)
            max_loss_percent = max_loss / self.portfolio_value
        else:
            max_loss_percent = self.position_size_limit
        
        position_risk = PositionRisk(
            symbol=symbol,
            position_size=size,
            risk_amount=risk_amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_loss_percent=max_loss_percent,
            current_risk_score=self._calculate_position_risk_score(symbol, size, entry_price)
        )
        
        self.positions[symbol] = position_risk
        logger.info(f"Position added: {symbol}, size: {size}, risk: {max_loss_percent:.2%}")
    
    def remove_position(self, symbol: str):
        """Remove a position from the portfolio"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Position removed: {symbol}")
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        # Total exposure
        total_exposure = sum(pos.risk_amount for pos in self.positions.values())
        exposure_ratio = total_exposure / self.portfolio_value
        
        # Position risk
        position_risk = sum(pos.max_loss_percent for pos in self.positions.values())
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk()
        
        # Volatility risk (from parent class)
        volatility_risk = self.current_volatility
        
        # News risk (from parent class)
        news_risk = self.market_event_risk
        
        # Drawdown risk
        drawdown_risk = self._calculate_drawdown_risk()
        
        # Overall risk score (weighted combination)
        weights = {
            'exposure': 0.25,
            'position': 0.20,
            'correlation': 0.15,
            'volatility': 0.15,
            'news': 0.15,
            'drawdown': 0.10
        }
        
        overall_risk_score = (
            exposure_ratio * weights['exposure'] +
            position_risk * weights['position'] +
            correlation_risk * weights['correlation'] +
            volatility_risk * weights['volatility'] +
            news_risk * weights['news'] +
            drawdown_risk * weights['drawdown']
        )
        
        # Determine risk level
        if overall_risk_score >= 0.8:
            risk_level = 'critical'
        elif overall_risk_score >= 0.6:
            risk_level = 'high'
        elif overall_risk_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        risk_metrics = RiskMetrics(
            total_exposure=total_exposure,
            position_risk=position_risk,
            correlation_risk=correlation_risk,
            volatility_risk=volatility_risk,
            news_risk=news_risk,
            drawdown_risk=drawdown_risk,
            overall_risk_score=overall_risk_score,
            risk_level=risk_level
        )
        
        # Store in history
        self.risk_history.append({
            'timestamp': datetime.now(),
            'metrics': risk_metrics
        })
        
        # Keep only last 1000 records
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
        
        return risk_metrics
    
    def _calculate_position_risk_score(self, symbol: str, size: float, price: float) -> float:
        """Calculate risk score for individual position"""
        position_value = abs(size * price)
        position_ratio = position_value / self.portfolio_value
        
        # Base risk from position size
        size_risk = min(position_ratio / self.max_position_size, 1.0)
        
        # Additional risk factors could be added here
        # (symbol-specific volatility, liquidity, etc.)
        
        return size_risk
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk"""
        if len(self.positions) < 2:
            return 0.0
        
        # Simplified correlation risk calculation
        # In production, this would use actual correlation data
        symbols = list(self.positions.keys())
        
        # Simulate correlation (in production, use historical data)
        correlations = []
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Simulate correlation between crypto pairs
                if 'BTC' in symbol1 and 'BTC' in symbol2:
                    corr = 0.8  # High correlation
                elif any(crypto in symbol1 and crypto in symbol2 
                        for crypto in ['ETH', 'ADA', 'DOT']):
                    corr = 0.6  # Medium correlation
                else:
                    corr = 0.3  # Low correlation
                
                correlations.append(corr)
        
        if correlations:
            avg_correlation = np.mean(correlations)
            # Higher correlation means higher risk
            return avg_correlation
        
        return 0.0
    
    def _calculate_drawdown_risk(self) -> float:
        """Calculate current drawdown risk"""
        if len(self.drawdown_history) < 2:
            return 0.0
        
        # Get recent drawdowns
        recent_drawdowns = [entry['drawdown'] for entry in self.drawdown_history[-30:]]
        current_drawdown = abs(min(recent_drawdowns, default=0))
        
        # Normalize to risk score
        return min(current_drawdown / self.max_portfolio_risk, 1.0)
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: Optional[float] = None) -> float:
        """Calculate optimal position size based on risk parameters"""
        
        # Base position size from portfolio risk
        base_risk_amount = self.portfolio_value * self.position_size_limit
        
        if stop_loss:
            # Calculate position size based on stop loss
            risk_per_unit = abs(entry_price - stop_loss)
            max_units = base_risk_amount / risk_per_unit
        else:
            # Use default risk without stop loss
            max_units = base_risk_amount / entry_price
        
        # Apply risk multiplier adjustments
        adjusted_units = max_units * self.risk_multiplier
        
        # Check portfolio limits
        max_position_value = self.portfolio_value * self.max_position_size
        max_units_by_value = max_position_value / entry_price
        
        # Use the smaller of the two limits
        final_units = min(adjusted_units, max_units_by_value)
        
        logger.info(f"Position size calculation for {symbol}: {final_units:.6f} units")
        return final_units
    
    def should_reduce_exposure(self) -> bool:
        """Check if overall exposure should be reduced"""
        risk_metrics = self.calculate_portfolio_risk()
        
        # Emergency stop conditions
        if (risk_metrics.overall_risk_score >= 0.9 or 
            risk_metrics.drawdown_risk >= 0.8 or
            self.emergency_stop):
            return True
        
        # Standard risk conditions from parent class
        return self.should_halt_trading()
    
    def adjust_risk_multiplier(self, market_conditions: Dict[str, Any]):
        """Dynamically adjust risk multiplier based on market conditions"""
        
        # Base multiplier
        multiplier = 1.0
        
        # Adjust based on volatility
        if self.current_volatility > 0.05:  # High volatility
            multiplier *= 0.7
        elif self.current_volatility < 0.02:  # Low volatility
            multiplier *= 1.2
        
        # Adjust based on news impact
        if self.market_event_risk > 0.7:  # High news impact
            multiplier *= 0.5
        
        # Adjust based on portfolio performance
        if len(self.drawdown_history) > 5:
            recent_performance = np.mean([entry['drawdown'] 
                                        for entry in self.drawdown_history[-5:]])
            if recent_performance < -0.03:  # Losing streak
                multiplier *= 0.8
            elif recent_performance > 0.02:  # Winning streak
                multiplier *= 1.1
        
        # Bounds checking
        self.risk_multiplier = max(0.1, min(2.0, multiplier))
        
        logger.debug(f"Risk multiplier adjusted to: {self.risk_multiplier:.2f}")
    
    def set_emergency_stop(self, enabled: bool, reason: str = ""):
        """Enable/disable emergency stop"""
        self.emergency_stop = enabled
        if enabled:
            logger.warning(f"Emergency stop activated: {reason}")
        else:
            logger.info("Emergency stop deactivated")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        risk_metrics = self.calculate_portfolio_risk()
        
        report = {
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'risk_metrics': {
                'overall_score': risk_metrics.overall_risk_score,
                'risk_level': risk_metrics.risk_level,
                'total_exposure': risk_metrics.total_exposure,
                'position_risk': risk_metrics.position_risk,
                'correlation_risk': risk_metrics.correlation_risk,
                'volatility_risk': risk_metrics.volatility_risk,
                'news_risk': risk_metrics.news_risk,
                'drawdown_risk': risk_metrics.drawdown_risk
            },
            'positions': {
                symbol: {
                    'size': pos.position_size,
                    'risk_amount': pos.risk_amount,
                    'max_loss_percent': pos.max_loss_percent,
                    'risk_score': pos.current_risk_score
                }
                for symbol, pos in self.positions.items()
            },
            'risk_controls': {
                'risk_multiplier': self.risk_multiplier,
                'emergency_stop': self.emergency_stop,
                'should_halt_trading': self.should_halt_trading(),
                'should_reduce_exposure': self.should_reduce_exposure()
            },
            'limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_size': self.max_position_size,
                'position_size_limit': self.position_size_limit
            }
        }
        
        return report
    
    def get_position_recommendations(self) -> Dict[str, str]:
        """Get position-specific recommendations"""
        recommendations = {}
        risk_metrics = self.calculate_portfolio_risk()
        
        for symbol, position in self.positions.items():
            if position.current_risk_score > 0.8:
                recommendations[symbol] = "REDUCE: High individual risk"
            elif risk_metrics.overall_risk_score > 0.7 and position.max_loss_percent > 0.02:
                recommendations[symbol] = "MONITOR: Portfolio risk elevated"
            elif position.stop_loss is None and position.max_loss_percent > 0.01:
                recommendations[symbol] = "SET_STOP_LOSS: No stop loss protection"
            else:
                recommendations[symbol] = "HOLD: Risk within limits"
        
        return recommendations
    
    def update_correlation_matrix(self, price_data: Dict[str, List[float]]):
        """Update correlation matrix with latest price data"""
        if len(price_data) < 2:
            return
        
        # Convert to DataFrame for correlation calculation
        df = pd.DataFrame(price_data)
        
        if len(df) > 10:  # Need sufficient data points
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            self.correlation_matrix = returns.corr().to_dict()
            
            logger.debug("Correlation matrix updated")

# Global enhanced risk manager instance
enhanced_risk_manager = None

def get_enhanced_risk_manager() -> EnhancedRiskManager:
    """Get the global enhanced risk manager instance"""
    global enhanced_risk_manager
    if enhanced_risk_manager is None:
        enhanced_risk_manager = EnhancedRiskManager()
    return enhanced_risk_manager