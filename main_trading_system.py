"""
Master Trading System Controller
Integrates ALL existing modules into a cohesive trading system
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta

# Import existing modules
from gut_trader import IntuitiveDecisionCore
from external_risk_manager import ExternalRiskManager
from reinforcement_learning_core import ReinforcementLearningCore
from omni_persona import OmniPersona
from global_sentiment import GlobalSentimentIntegrator
from emotional_responder import EmotionalResponseEngine
from final_expansion_for_advanced_predictions import FinalExpansionForAdvancedPredictions
from vision_live import VisionLiveAnalyzer
from live_visual_analysis import LiveVisualAnalysis
from social_pulse import SocialPulseMonitor
from group_behavior import GroupBehaviorAnalyzer

# Import new system components
from config import get_config

logger = logging.getLogger(__name__)

class MainTradingSystem:
    """
    Master controller that orchestrates all trading system components
    """
    
    def __init__(self):
        self.config = get_config()
        self.running = False
        self.last_decision_time = None
        
        # Initialize existing core modules
        self.gut_trader = IntuitiveDecisionCore()
        self.risk_manager = ExternalRiskManager(
            volatility_threshold=self.config.VOLATILITY_THRESHOLD,
            news_impact_weight=self.config.NEWS_IMPACT_WEIGHT
        )
        self.rl_core = ReinforcementLearningCore(
            actions=['buy', 'sell', 'hold'],
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.1
        )
        self.omni_persona = OmniPersona(self.config.get_persona_config()['default_persona'])
        self.sentiment_integrator = GlobalSentimentIntegrator()
        self.emotional_responder = EmotionalResponseEngine()
        
        # Initialize prediction systems
        self.advanced_predictor = FinalExpansionForAdvancedPredictions()
        
        # Initialize analysis modules
        try:
            self.vision_analyzer = VisionLiveAnalyzer()
            self.visual_analysis = LiveVisualAnalysis()
        except Exception as e:
            logger.warning(f"Visual analysis modules not available: {e}")
            self.vision_analyzer = None
            self.visual_analysis = None
        
        try:
            self.social_pulse = SocialPulseMonitor()
            self.group_behavior = GroupBehaviorAnalyzer()
        except Exception as e:
            logger.warning(f"Social analysis modules not available: {e}")
            self.social_pulse = None
            self.group_behavior = None
        
        # Trading state
        self.current_positions = {}
        self.trading_history = []
        self.market_data_cache = {}
        self.sentiment_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info("Main Trading System initialized with all modules")
    
    async def start(self):
        """Start the main trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return
        
        logger.info("Starting Main Trading System...")
        self.running = True
        
        # Start main trading loop
        await self._main_trading_loop()
    
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping Main Trading System...")
        self.running = False
    
    async def _main_trading_loop(self):
        """Main trading loop that coordinates all modules"""
        while self.running:
            try:
                # Get current market state
                market_state = await self._get_market_state()
                
                # Update all modules with current data
                await self._update_modules(market_state)
                
                # Make trading decision
                decision = await self._make_trading_decision(market_state)
                
                # Execute decision if valid
                if decision and decision['action'] != 'hold':
                    await self._execute_decision(decision, market_state)
                
                # Update performance tracking
                self._update_performance_metrics()
                
                # Log system status
                self._log_system_status(market_state, decision)
                
                # Wait for next iteration
                await asyncio.sleep(self.config.UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _get_market_state(self) -> Dict[str, Any]:
        """Collect current market state from all sources"""
        market_state = {
            'timestamp': datetime.now(),
            'prices': {},
            'volumes': {},
            'technical_indicators': {},
            'sentiment_scores': {},
            'news_impact': 0.0,
            'volatility': 0.0
        }
        
        # This would normally fetch from data_manager
        # For now, simulate market data
        for symbol in self.config.SYMBOLS:
            market_state['prices'][symbol] = np.random.uniform(40000, 50000)  # Simulated price
            market_state['volumes'][symbol] = np.random.uniform(1000, 5000)
            market_state['technical_indicators'][symbol] = {
                'rsi': np.random.uniform(30, 70),
                'macd': np.random.uniform(-100, 100),
                'bollinger_bands': {
                    'upper': np.random.uniform(45000, 55000),
                    'middle': np.random.uniform(40000, 50000),
                    'lower': np.random.uniform(35000, 45000)
                }
            }
        
        # Calculate overall market volatility
        if len(self.market_data_cache) > 0:
            prices = [state['prices'].get('BTCUSDT', 0) for state in self.market_data_cache.values()]
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                market_state['volatility'] = np.std(returns)
        
        # Cache the market state
        self.market_data_cache[market_state['timestamp']] = market_state
        
        # Keep only last 100 states
        if len(self.market_data_cache) > 100:
            oldest_key = min(self.market_data_cache.keys())
            del self.market_data_cache[oldest_key]
        
        return market_state
    
    async def _update_modules(self, market_state: Dict[str, Any]):
        """Update all modules with current market data"""
        
        # Update risk manager
        prices = list(market_state['prices'].values())
        if len(prices) > 0:
            self.risk_manager.update_volatility(np.array(prices))
            self.risk_manager.update_news_impact(market_state['news_impact'])
        
        # Update sentiment integrator
        sentiment_score = np.random.choice(['positive', 'negative', 'neutral'])  # Simulated
        self.sentiment_integrator.integrate_sentiment(sentiment_score)
        
        # Update persona based on market conditions
        market_condition = 'volatile' if market_state['volatility'] > 0.05 else 'stable'
        self.omni_persona.adjust_behavior(market_condition)
        
        # Update advanced predictor
        if len(prices) > 0:
            prediction_data = {
                'sentiment': 0.5,  # Normalized sentiment
                'volatility': market_state['volatility'],
                'price_change': np.random.uniform(-0.1, 0.1)  # Simulated price change
            }
            self.advanced_predictor.process_market_data(prediction_data)
    
    async def _make_trading_decision(self, market_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Combine all module inputs to make a trading decision"""
        
        # Check risk manager first
        risk_signal = self.risk_manager.generate_signal()
        if risk_signal['action'] == 'HOLD':
            return {
                'action': 'hold',
                'reason': risk_signal['reason'],
                'confidence': 0.0,
                'symbol': 'BTCUSDT'
            }
        
        # Get gut trader decision
        pattern_rarity = np.random.uniform(0, 1)
        memory_match_score = np.random.uniform(0, 1)
        emotional_pressure = np.random.uniform(0, 1)
        gut_decision = self.gut_trader.decide(pattern_rarity, memory_match_score, emotional_pressure)
        
        # Get RL decision
        current_state = f"price_{int(market_state['prices'].get('BTCUSDT', 0) / 1000)}"
        rl_decision = self.rl_core.select_action(current_state)
        
        # Get persona influence
        persona = self.omni_persona.get_personality()
        risk_tolerance = persona['risk_tolerance']
        
        # Get sentiment influence
        dominant_sentiment = self.sentiment_integrator.dominant_sentiment()
        
        # Get advanced predictor decision
        if len(self.advanced_predictor.market_data) > 0:
            # This would use the actual prediction method
            prediction_confidence = np.random.uniform(0.4, 0.9)
        else:
            prediction_confidence = 0.5
        
        # Combine all decisions with weights
        decision_weights = {
            'gut': 0.3,
            'rl': 0.25,
            'persona': 0.15,
            'sentiment': 0.15,
            'prediction': 0.15
        }
        
        # Score each action
        action_scores = {'buy': 0, 'sell': 0, 'hold': 0}
        
        # Gut trader influence
        if gut_decision == 'buy':
            action_scores['buy'] += decision_weights['gut']
        elif gut_decision == 'sell':
            action_scores['sell'] += decision_weights['gut']
        else:
            action_scores['hold'] += decision_weights['gut']
        
        # RL influence
        action_scores[rl_decision] += decision_weights['rl']
        
        # Sentiment influence
        if dominant_sentiment == 'positive':
            action_scores['buy'] += decision_weights['sentiment']
        elif dominant_sentiment == 'negative':
            action_scores['sell'] += decision_weights['sentiment']
        else:
            action_scores['hold'] += decision_weights['sentiment']
        
        # Prediction influence
        if prediction_confidence > 0.7:
            action_scores['buy'] += decision_weights['prediction']
        elif prediction_confidence < 0.3:
            action_scores['sell'] += decision_weights['prediction']
        else:
            action_scores['hold'] += decision_weights['prediction']
        
        # Persona risk adjustment
        if risk_tolerance < 0.3:  # Conservative
            action_scores['hold'] += decision_weights['persona']
        elif risk_tolerance > 0.7:  # Aggressive
            # Add to highest non-hold action
            if action_scores['buy'] > action_scores['sell']:
                action_scores['buy'] += decision_weights['persona']
            else:
                action_scores['sell'] += decision_weights['persona']
        
        # Select final action
        final_action = max(action_scores, key=action_scores.get)
        confidence = action_scores[final_action]
        
        # Only execute if confidence is above threshold
        min_confidence = 0.6
        if confidence < min_confidence:
            final_action = 'hold'
        
        decision = {
            'action': final_action,
            'confidence': confidence,
            'symbol': 'BTCUSDT',  # For now, trade only BTC
            'reasoning': {
                'gut_decision': gut_decision,
                'rl_decision': rl_decision,
                'sentiment': dominant_sentiment,
                'risk_tolerance': risk_tolerance,
                'prediction_confidence': prediction_confidence,
                'action_scores': action_scores
            },
            'timestamp': datetime.now()
        }
        
        self.last_decision_time = decision['timestamp']
        return decision
    
    async def _execute_decision(self, decision: Dict[str, Any], market_state: Dict[str, Any]):
        """Execute trading decision (placeholder for actual execution)"""
        logger.info(f"Executing decision: {decision['action']} {decision['symbol']} "
                   f"with confidence {decision['confidence']:.2f}")
        
        # This is where we would interface with market_connectors.py
        # For now, just log and track the decision
        
        trade_record = {
            'timestamp': decision['timestamp'],
            'symbol': decision['symbol'],
            'action': decision['action'],
            'price': market_state['prices'].get(decision['symbol'], 0),
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning']
        }
        
        self.trading_history.append(trade_record)
        self.performance_metrics['total_trades'] += 1
        
        # Update RL with simulated reward
        current_state = f"price_{int(market_state['prices'].get(decision['symbol'], 0) / 1000)}"
        next_state = current_state  # Simplified
        reward = np.random.uniform(-1, 1)  # Simulated reward
        
        self.rl_core.update(current_state, decision['action'], reward, next_state)
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        if len(self.trading_history) < 2:
            return
        
        # Calculate basic metrics (simplified)
        recent_trades = self.trading_history[-10:]  # Last 10 trades
        
        # Simulate PnL calculation
        for trade in recent_trades:
            simulated_pnl = np.random.uniform(-50, 100)  # Simulated profit/loss
            self.performance_metrics['total_pnl'] += simulated_pnl
            
            if simulated_pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
    
    def _log_system_status(self, market_state: Dict[str, Any], decision: Optional[Dict[str, Any]]):
        """Log current system status"""
        if self.last_decision_time and (datetime.now() - self.last_decision_time).seconds < 60:
            return  # Don't log too frequently
        
        btc_price = market_state['prices'].get('BTCUSDT', 0)
        volatility = market_state['volatility']
        
        status_msg = (
            f"System Status - BTC: ${btc_price:.2f}, "
            f"Volatility: {volatility:.4f}, "
            f"Trades: {self.performance_metrics['total_trades']}, "
            f"PnL: ${self.performance_metrics['total_pnl']:.2f}"
        )
        
        if decision:
            status_msg += f", Decision: {decision['action']} ({decision['confidence']:.2f})"
        
        logger.info(status_msg)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'running': self.running,
            'modules_loaded': {
                'gut_trader': self.gut_trader is not None,
                'risk_manager': self.risk_manager is not None,
                'rl_core': self.rl_core is not None,
                'omni_persona': self.omni_persona is not None,
                'sentiment_integrator': self.sentiment_integrator is not None,
                'advanced_predictor': self.advanced_predictor is not None,
                'vision_analyzer': self.vision_analyzer is not None,
                'social_pulse': self.social_pulse is not None,
            },
            'performance': self.performance_metrics,
            'last_decision': self.last_decision_time,
            'active_positions': len(self.current_positions),
            'trading_history_count': len(self.trading_history)
        }

# Global trading system instance
trading_system = None

def get_trading_system() -> MainTradingSystem:
    """Get the global trading system instance"""
    global trading_system
    if trading_system is None:
        trading_system = MainTradingSystem()
    return trading_system

async def main():
    """Main entry point for the trading system"""
    import logging.config
    
    # Setup logging
    config = get_config()
    logging.config.dictConfig(config.get_logging_config())
    
    # Validate configuration
    if not config.validate_config():
        logger.error("Configuration validation failed. Exiting.")
        return
    
    # Start trading system
    system = get_trading_system()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        system.stop()
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())