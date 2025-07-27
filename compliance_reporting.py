"""
Compliance and Regulatory Reporting System for OmniBeing Trading Platform.
Handles financial compliance, regulatory reporting, and audit trail generation.
"""

import asyncio
import logging
import json
import csv
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import redis
import asyncpg
from production_config import ProductionConfig

class ReportType(Enum):
    """Types of compliance reports."""
    TRADE_REPORT = "trade_report"
    TAX_REPORT = "tax_report"
    RISK_REPORT = "risk_report"
    AML_REPORT = "aml_report"
    AUDIT_TRAIL = "audit_trail"
    POSITION_REPORT = "position_report"
    PNL_REPORT = "pnl_report"
    COMPLIANCE_SUMMARY = "compliance_summary"

class ComplianceLevel(Enum):
    """Compliance severity levels."""
    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"

class RegulatoryJurisdiction(Enum):
    """Regulatory jurisdictions."""
    SEC = "sec"  # US Securities and Exchange Commission
    CFTC = "cftc"  # US Commodity Futures Trading Commission
    FCA = "fca"  # UK Financial Conduct Authority
    ESMA = "esma"  # European Securities and Markets Authority
    MAS = "mas"  # Monetary Authority of Singapore
    FSA = "fsa"  # Financial Services Agency (Japan)

@dataclass
class TradeRecord:
    """Trade record for compliance reporting."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    value: float
    fees: float
    exchange: str
    strategy: str
    user_id: str
    settlement_date: datetime
    trade_type: str
    regulatory_flags: List[str]

@dataclass
class ComplianceEvent:
    """Compliance event record."""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: ComplianceLevel
    description: str
    affected_trades: List[str]
    user_id: Optional[str]
    regulatory_impact: List[RegulatoryJurisdiction]
    remediation_required: bool
    remediation_deadline: Optional[datetime]
    status: str

@dataclass
class TaxCalculation:
    """Tax calculation details."""
    calculation_id: str
    user_id: str
    period_start: datetime
    period_end: datetime
    total_gains: float
    total_losses: float
    net_gains: float
    short_term_gains: float
    long_term_gains: float
    wash_sale_adjustments: float
    tax_liability: float
    jurisdiction: str

@dataclass
class RiskMetrics:
    """Risk assessment metrics."""
    timestamp: datetime
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    max_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    concentration_risk: float
    liquidity_risk: float

@dataclass
class AMLAlert:
    """Anti-Money Laundering alert."""
    alert_id: str
    timestamp: datetime
    user_id: str
    alert_type: str
    severity: str
    description: str
    triggering_transactions: List[str]
    investigation_status: str
    investigator: Optional[str]
    resolution: Optional[str]
    reported_to_authorities: bool

class ComplianceReporting:
    """
    Comprehensive compliance and regulatory reporting system.
    Handles trade reporting, tax calculations, risk monitoring, and audit trails.
    """
    
    def __init__(self, config: ProductionConfig):
        """
        Initialize compliance reporting system.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Database and cache
        self.db_pool = None
        self.redis_client = None
        
        # Compliance configuration
        self.jurisdictions = [RegulatoryJurisdiction.SEC, RegulatoryJurisdiction.CFTC]
        self.reporting_thresholds = {
            'large_trader_threshold': 2000000,  # $2M for large trader reporting
            'position_limit_threshold': 5000000,  # $5M position limit
            'daily_volume_threshold': 10000000,  # $10M daily volume threshold
            'concentration_threshold': 0.1,  # 10% concentration limit
            'var_threshold': 0.05,  # 5% VaR threshold
            'leverage_threshold': 10.0  # 10x leverage threshold
        }
        
        # Tax configuration
        self.tax_rates = {
            'short_term_rate': 0.37,  # 37% short-term capital gains
            'long_term_rate': 0.20,   # 20% long-term capital gains
            'holding_period_threshold': 365  # Days for long-term treatment
        }
        
        # AML configuration
        self.aml_thresholds = {
            'suspicious_volume': 1000000,  # $1M suspicious volume
            'rapid_transaction_count': 100,  # 100 trades in short period
            'unusual_pattern_threshold': 3.0,  # 3 standard deviations
            'cross_border_threshold': 10000  # $10K cross-border reporting
        }
        
        # Compliance state
        self.compliance_events: List[ComplianceEvent] = []
        self.aml_alerts: List[AMLAlert] = []
        self.pending_reports: List[Dict] = []
        
        # Report generation
        self.report_templates = self._setup_report_templates()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup compliance logger."""
        logger = logging.getLogger('compliance')
        logger.setLevel(logging.INFO)
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler('logs/compliance.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [COMPLIANCE] %(message)s')
        )
        logger.addHandler(file_handler)
        
        # Separate audit trail handler
        audit_handler = logging.FileHandler('logs/audit_trail.log')
        audit_handler.setFormatter(
            logging.Formatter('%(asctime)s - AUDIT - %(message)s')
        )
        audit_logger = logging.getLogger('audit')
        audit_logger.addHandler(audit_handler)
        
        return logger
    
    def _setup_report_templates(self) -> Dict[str, Dict]:
        """Setup report templates."""
        return {
            'trade_report': {
                'filename': 'trade_report_{date}.csv',
                'columns': ['trade_id', 'timestamp', 'symbol', 'side', 'quantity', 
                          'price', 'value', 'fees', 'exchange', 'strategy'],
                'regulatory_fields': ['settlement_date', 'trade_type', 'counterparty']
            },
            'tax_report': {
                'filename': 'tax_report_{year}_{user_id}.pdf',
                'sections': ['summary', 'realized_gains', 'unrealized_gains', 
                           'wash_sales', 'cost_basis'],
                'forms': ['Schedule D', 'Form 8949']
            },
            'risk_report': {
                'filename': 'risk_report_{date}.pdf',
                'metrics': ['var', 'expected_shortfall', 'concentration', 
                          'liquidity', 'leverage'],
                'frequency': 'daily'
            },
            'aml_report': {
                'filename': 'aml_report_{date}.csv',
                'columns': ['alert_id', 'timestamp', 'user_id', 'alert_type', 
                          'severity', 'status'],
                'regulatory_filing': True
            }
        }
    
    async def initialize(self):
        """Initialize compliance reporting system."""
        try:
            # Initialize Redis connection
            redis_url = self.config.get_redis_url()
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Initialize database connection
            db_url = self.config.get_database_url()
            self.db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=5)
            
            # Create compliance tables
            await self._create_compliance_tables()
            
            # Start compliance monitoring
            asyncio.create_task(self._compliance_monitoring_loop())
            asyncio.create_task(self._aml_monitoring_loop())
            asyncio.create_task(self._report_generation_loop())
            
            self.logger.info("Compliance reporting system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize compliance system: {e}")
            raise
    
    async def _create_compliance_tables(self):
        """Create database tables for compliance data."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Trade records table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS trade_records (
                            trade_id VARCHAR PRIMARY KEY,
                            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                            symbol VARCHAR(20) NOT NULL,
                            side VARCHAR(10) NOT NULL,
                            quantity DECIMAL(18,8) NOT NULL,
                            price DECIMAL(18,8) NOT NULL,
                            value DECIMAL(18,8) NOT NULL,
                            fees DECIMAL(18,8) NOT NULL,
                            exchange VARCHAR(50) NOT NULL,
                            strategy VARCHAR(100),
                            user_id VARCHAR(100),
                            settlement_date DATE,
                            trade_type VARCHAR(50),
                            regulatory_flags JSONB,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    
                    # Compliance events table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS compliance_events (
                            event_id VARCHAR PRIMARY KEY,
                            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                            event_type VARCHAR(100) NOT NULL,
                            severity VARCHAR(20) NOT NULL,
                            description TEXT NOT NULL,
                            affected_trades JSONB,
                            user_id VARCHAR(100),
                            regulatory_impact JSONB,
                            remediation_required BOOLEAN DEFAULT FALSE,
                            remediation_deadline TIMESTAMP,
                            status VARCHAR(50) DEFAULT 'open',
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    
                    # Tax calculations table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS tax_calculations (
                            calculation_id VARCHAR PRIMARY KEY,
                            user_id VARCHAR(100) NOT NULL,
                            period_start DATE NOT NULL,
                            period_end DATE NOT NULL,
                            total_gains DECIMAL(18,8),
                            total_losses DECIMAL(18,8),
                            net_gains DECIMAL(18,8),
                            short_term_gains DECIMAL(18,8),
                            long_term_gains DECIMAL(18,8),
                            wash_sale_adjustments DECIMAL(18,8),
                            tax_liability DECIMAL(18,8),
                            jurisdiction VARCHAR(10),
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    
                    # AML alerts table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS aml_alerts (
                            alert_id VARCHAR PRIMARY KEY,
                            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                            user_id VARCHAR(100) NOT NULL,
                            alert_type VARCHAR(100) NOT NULL,
                            severity VARCHAR(20) NOT NULL,
                            description TEXT NOT NULL,
                            triggering_transactions JSONB,
                            investigation_status VARCHAR(50) DEFAULT 'pending',
                            investigator VARCHAR(100),
                            resolution TEXT,
                            reported_to_authorities BOOLEAN DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    
                    # Risk metrics table
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS risk_metrics (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                            var_95 DECIMAL(10,6),
                            var_99 DECIMAL(10,6),
                            expected_shortfall DECIMAL(10,6),
                            max_drawdown DECIMAL(10,6),
                            volatility DECIMAL(10,6),
                            beta DECIMAL(10,6),
                            sharpe_ratio DECIMAL(10,6),
                            concentration_risk DECIMAL(10,6),
                            liquidity_risk DECIMAL(10,6),
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_records_timestamp ON trade_records(timestamp)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_records_user ON trade_records(user_id)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_compliance_events_timestamp ON compliance_events(timestamp)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_aml_alerts_timestamp ON aml_alerts(timestamp)")
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp)")
                    
                    self.logger.info("Compliance database tables created")
                    
        except Exception as e:
            self.logger.error(f"Error creating compliance tables: {e}")
    
    async def record_trade(self, trade_data: Dict[str, Any]) -> str:
        """Record a trade for compliance tracking."""
        try:
            # Create trade record
            trade_record = TradeRecord(
                trade_id=trade_data['trade_id'],
                timestamp=trade_data['timestamp'],
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                price=trade_data['price'],
                value=trade_data['quantity'] * trade_data['price'],
                fees=trade_data.get('fees', 0.0),
                exchange=trade_data['exchange'],
                strategy=trade_data.get('strategy', 'manual'),
                user_id=trade_data.get('user_id', 'system'),
                settlement_date=trade_data.get('settlement_date', trade_data['timestamp'].date()),
                trade_type=trade_data.get('trade_type', 'equity'),
                regulatory_flags=trade_data.get('regulatory_flags', [])
            )
            
            # Store in database
            await self._store_trade_record(trade_record)
            
            # Check compliance rules
            await self._check_trade_compliance(trade_record)
            
            # Log audit trail
            audit_logger = logging.getLogger('audit')
            audit_logger.info(f"TRADE_RECORDED: {trade_record.trade_id} - {trade_record.symbol} {trade_record.side} {trade_record.quantity} @ {trade_record.price}")
            
            return trade_record.trade_id
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
            raise
    
    async def _store_trade_record(self, trade_record: TradeRecord):
        """Store trade record in database."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO trade_records 
                        (trade_id, timestamp, symbol, side, quantity, price, value,
                         fees, exchange, strategy, user_id, settlement_date, 
                         trade_type, regulatory_flags)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                        ON CONFLICT (trade_id) DO NOTHING
                    """, trade_record.trade_id, trade_record.timestamp, trade_record.symbol,
                        trade_record.side, trade_record.quantity, trade_record.price,
                        trade_record.value, trade_record.fees, trade_record.exchange,
                        trade_record.strategy, trade_record.user_id, trade_record.settlement_date,
                        trade_record.trade_type, json.dumps(trade_record.regulatory_flags))
        except Exception as e:
            self.logger.error(f"Error storing trade record: {e}")
    
    async def _check_trade_compliance(self, trade_record: TradeRecord):
        """Check trade against compliance rules."""
        try:
            # Check position limits
            await self._check_position_limits(trade_record)
            
            # Check large trader reporting
            await self._check_large_trader_threshold(trade_record)
            
            # Check unusual activity
            await self._check_unusual_activity(trade_record)
            
            # Check market manipulation patterns
            await self._check_market_manipulation(trade_record)
            
        except Exception as e:
            self.logger.error(f"Error checking trade compliance: {e}")
    
    async def _check_position_limits(self, trade_record: TradeRecord):
        """Check if trade violates position limits."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Calculate current position value for user and symbol
                    result = await conn.fetchval("""
                        SELECT COALESCE(SUM(
                            CASE 
                                WHEN side = 'buy' THEN value
                                WHEN side = 'sell' THEN -value
                            END
                        ), 0)
                        FROM trade_records 
                        WHERE user_id = $1 AND symbol = $2
                    """, trade_record.user_id, trade_record.symbol)
                    
                    position_value = float(result) if result else 0.0
                    
                    if abs(position_value) > self.reporting_thresholds['position_limit_threshold']:
                        await self._create_compliance_event(
                            event_type="position_limit_violation",
                            severity=ComplianceLevel.VIOLATION,
                            description=f"Position limit exceeded for {trade_record.symbol}: ${position_value:,.2f}",
                            affected_trades=[trade_record.trade_id],
                            user_id=trade_record.user_id,
                            remediation_required=True
                        )
                        
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
    
    async def _check_large_trader_threshold(self, trade_record: TradeRecord):
        """Check if user exceeds large trader reporting threshold."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Check daily trading volume for user
                    result = await conn.fetchval("""
                        SELECT COALESCE(SUM(value), 0)
                        FROM trade_records 
                        WHERE user_id = $1 
                        AND DATE(timestamp) = CURRENT_DATE
                    """, trade_record.user_id)
                    
                    daily_volume = float(result) if result else 0.0
                    
                    if daily_volume > self.reporting_thresholds['large_trader_threshold']:
                        await self._create_compliance_event(
                            event_type="large_trader_threshold",
                            severity=ComplianceLevel.WARNING,
                            description=f"Large trader threshold exceeded: ${daily_volume:,.2f}",
                            affected_trades=[trade_record.trade_id],
                            user_id=trade_record.user_id,
                            regulatory_impact=[RegulatoryJurisdiction.SEC]
                        )
                        
        except Exception as e:
            self.logger.error(f"Error checking large trader threshold: {e}")
    
    async def _check_unusual_activity(self, trade_record: TradeRecord):
        """Check for unusual trading activity patterns."""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Check for rapid trading
                    result = await conn.fetchval("""
                        SELECT COUNT(*)
                        FROM trade_records 
                        WHERE user_id = $1 
                        AND timestamp >= NOW() - INTERVAL '1 hour'
                    """, trade_record.user_id)
                    
                    trade_count = int(result) if result else 0
                    
                    if trade_count > self.aml_thresholds['rapid_transaction_count']:
                        await self._create_aml_alert(
                            user_id=trade_record.user_id,
                            alert_type="rapid_trading",
                            severity="medium",
                            description=f"Rapid trading detected: {trade_count} trades in 1 hour",
                            triggering_transactions=[trade_record.trade_id]
                        )
                        
        except Exception as e:
            self.logger.error(f"Error checking unusual activity: {e}")
    
    async def _check_market_manipulation(self, trade_record: TradeRecord):
        """Check for potential market manipulation patterns."""
        try:
            # Simplified market manipulation detection
            # In practice, this would involve complex statistical analysis
            
            if trade_record.value > 1000000:  # Large order
                await self._create_compliance_event(
                    event_type="large_order_review",
                    severity=ComplianceLevel.INFO,
                    description=f"Large order requiring review: ${trade_record.value:,.2f}",
                    affected_trades=[trade_record.trade_id],
                    user_id=trade_record.user_id
                )
                
        except Exception as e:
            self.logger.error(f"Error checking market manipulation: {e}")
    
    async def _create_compliance_event(self,
                                     event_type: str,
                                     severity: ComplianceLevel,
                                     description: str,
                                     affected_trades: List[str],
                                     user_id: Optional[str] = None,
                                     regulatory_impact: List[RegulatoryJurisdiction] = None,
                                     remediation_required: bool = False):
        """Create a compliance event."""
        try:
            event = ComplianceEvent(
                event_id=f"CE_{int(datetime.now().timestamp())}_{len(self.compliance_events)}",
                timestamp=datetime.now(),
                event_type=event_type,
                severity=severity,
                description=description,
                affected_trades=affected_trades,
                user_id=user_id,
                regulatory_impact=regulatory_impact or [],
                remediation_required=remediation_required,
                remediation_deadline=datetime.now() + timedelta(days=30) if remediation_required else None,
                status="open"
            )
            
            self.compliance_events.append(event)
            
            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO compliance_events 
                        (event_id, timestamp, event_type, severity, description,
                         affected_trades, user_id, regulatory_impact, remediation_required,
                         remediation_deadline, status)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, event.event_id, event.timestamp, event.event_type,
                        event.severity.value, event.description, json.dumps(event.affected_trades),
                        event.user_id, json.dumps([j.value for j in event.regulatory_impact]),
                        event.remediation_required, event.remediation_deadline, event.status)
            
            self.logger.warning(f"Compliance event created: {event.event_id} - {event.description}")
            
        except Exception as e:
            self.logger.error(f"Error creating compliance event: {e}")
    
    async def _create_aml_alert(self,
                              user_id: str,
                              alert_type: str,
                              severity: str,
                              description: str,
                              triggering_transactions: List[str]):
        """Create an AML alert."""
        try:
            alert = AMLAlert(
                alert_id=f"AML_{int(datetime.now().timestamp())}_{len(self.aml_alerts)}",
                timestamp=datetime.now(),
                user_id=user_id,
                alert_type=alert_type,
                severity=severity,
                description=description,
                triggering_transactions=triggering_transactions,
                investigation_status="pending",
                investigator=None,
                resolution=None,
                reported_to_authorities=False
            )
            
            self.aml_alerts.append(alert)
            
            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO aml_alerts 
                        (alert_id, timestamp, user_id, alert_type, severity,
                         description, triggering_transactions, investigation_status)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, alert.alert_id, alert.timestamp, alert.user_id,
                        alert.alert_type, alert.severity, alert.description,
                        json.dumps(alert.triggering_transactions), alert.investigation_status)
            
            self.logger.warning(f"AML alert created: {alert.alert_id} - {alert.description}")
            
        except Exception as e:
            self.logger.error(f"Error creating AML alert: {e}")
    
    async def calculate_taxes(self, user_id: str, year: int) -> TaxCalculation:
        """Calculate taxes for a user for a given year."""
        try:
            if not self.db_pool:
                raise ValueError("Database not available")
            
            async with self.db_pool.acquire() as conn:
                # Get all trades for the year
                trades = await conn.fetch("""
                    SELECT * FROM trade_records 
                    WHERE user_id = $1 
                    AND EXTRACT(YEAR FROM timestamp) = $2
                    ORDER BY timestamp
                """, user_id, year)
                
                # Calculate gains and losses
                total_gains = 0.0
                total_losses = 0.0
                short_term_gains = 0.0
                long_term_gains = 0.0
                wash_sale_adjustments = 0.0
                
                # Simplified tax calculation (FIFO method)
                position_queue = {}  # symbol -> [(quantity, price, date), ...]
                
                for trade in trades:
                    symbol = trade['symbol']
                    quantity = float(trade['quantity'])
                    price = float(trade['price'])
                    trade_date = trade['timestamp']
                    
                    if symbol not in position_queue:
                        position_queue[symbol] = []
                    
                    if trade['side'] == 'buy':
                        # Add to position
                        position_queue[symbol].append((quantity, price, trade_date))
                    else:
                        # Sell - calculate gains/losses
                        remaining_to_sell = quantity
                        
                        while remaining_to_sell > 0 and position_queue[symbol]:
                            buy_quantity, buy_price, buy_date = position_queue[symbol][0]
                            
                            sell_quantity = min(remaining_to_sell, buy_quantity)
                            gain_loss = sell_quantity * (price - buy_price)
                            
                            # Determine if short-term or long-term
                            holding_period = (trade_date - buy_date).days
                            
                            if gain_loss > 0:
                                total_gains += gain_loss
                                if holding_period >= self.tax_rates['holding_period_threshold']:
                                    long_term_gains += gain_loss
                                else:
                                    short_term_gains += gain_loss
                            else:
                                total_losses += abs(gain_loss)
                            
                            # Update position queue
                            if sell_quantity == buy_quantity:
                                position_queue[symbol].pop(0)
                            else:
                                position_queue[symbol][0] = (
                                    buy_quantity - sell_quantity,
                                    buy_price,
                                    buy_date
                                )
                            
                            remaining_to_sell -= sell_quantity
                
                # Calculate tax liability
                net_gains = total_gains - total_losses
                short_term_tax = short_term_gains * self.tax_rates['short_term_rate']
                long_term_tax = long_term_gains * self.tax_rates['long_term_rate']
                total_tax_liability = short_term_tax + long_term_tax
                
                # Create tax calculation
                calculation = TaxCalculation(
                    calculation_id=f"TAX_{user_id}_{year}_{int(datetime.now().timestamp())}",
                    user_id=user_id,
                    period_start=datetime(year, 1, 1),
                    period_end=datetime(year, 12, 31),
                    total_gains=total_gains,
                    total_losses=total_losses,
                    net_gains=net_gains,
                    short_term_gains=short_term_gains,
                    long_term_gains=long_term_gains,
                    wash_sale_adjustments=wash_sale_adjustments,
                    tax_liability=total_tax_liability,
                    jurisdiction="US"
                )
                
                # Store in database
                await conn.execute("""
                    INSERT INTO tax_calculations 
                    (calculation_id, user_id, period_start, period_end, total_gains,
                     total_losses, net_gains, short_term_gains, long_term_gains,
                     wash_sale_adjustments, tax_liability, jurisdiction)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (calculation_id) DO UPDATE SET
                        total_gains = $5, total_losses = $6, net_gains = $7,
                        short_term_gains = $8, long_term_gains = $9,
                        wash_sale_adjustments = $10, tax_liability = $11
                """, calculation.calculation_id, calculation.user_id,
                    calculation.period_start, calculation.period_end,
                    calculation.total_gains, calculation.total_losses,
                    calculation.net_gains, calculation.short_term_gains,
                    calculation.long_term_gains, calculation.wash_sale_adjustments,
                    calculation.tax_liability, calculation.jurisdiction)
                
                self.logger.info(f"Tax calculation completed for {user_id} ({year}): ${total_tax_liability:,.2f}")
                
                return calculation
                
        except Exception as e:
            self.logger.error(f"Error calculating taxes: {e}")
            raise
    
    async def generate_report(self, report_type: ReportType, parameters: Dict[str, Any]) -> str:
        """Generate a compliance report."""
        try:
            if report_type == ReportType.TRADE_REPORT:
                return await self._generate_trade_report(parameters)
            elif report_type == ReportType.TAX_REPORT:
                return await self._generate_tax_report(parameters)
            elif report_type == ReportType.RISK_REPORT:
                return await self._generate_risk_report(parameters)
            elif report_type == ReportType.AML_REPORT:
                return await self._generate_aml_report(parameters)
            elif report_type == ReportType.AUDIT_TRAIL:
                return await self._generate_audit_trail(parameters)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
                
        except Exception as e:
            self.logger.error(f"Error generating report {report_type}: {e}")
            raise
    
    async def _generate_trade_report(self, parameters: Dict[str, Any]) -> str:
        """Generate trade report."""
        try:
            start_date = parameters.get('start_date', datetime.now().date() - timedelta(days=30))
            end_date = parameters.get('end_date', datetime.now().date())
            user_id = parameters.get('user_id')
            
            if not self.db_pool:
                raise ValueError("Database not available")
            
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT trade_id, timestamp, symbol, side, quantity, price, 
                           value, fees, exchange, strategy, user_id
                    FROM trade_records 
                    WHERE DATE(timestamp) BETWEEN $1 AND $2
                """
                params = [start_date, end_date]
                
                if user_id:
                    query += " AND user_id = $3"
                    params.append(user_id)
                
                query += " ORDER BY timestamp"
                
                trades = await conn.fetch(query, *params)
                
                # Generate CSV report
                report_filename = f"reports/trade_report_{start_date}_{end_date}.csv"
                Path("reports").mkdir(exist_ok=True)
                
                with open(report_filename, 'w', newline='') as csvfile:
                    fieldnames = ['trade_id', 'timestamp', 'symbol', 'side', 'quantity', 
                                'price', 'value', 'fees', 'exchange', 'strategy', 'user_id']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for trade in trades:
                        writer.writerow(dict(trade))
                
                self.logger.info(f"Trade report generated: {report_filename}")
                return report_filename
                
        except Exception as e:
            self.logger.error(f"Error generating trade report: {e}")
            raise
    
    async def _generate_tax_report(self, parameters: Dict[str, Any]) -> str:
        """Generate tax report."""
        try:
            user_id = parameters['user_id']
            year = parameters['year']
            
            # Calculate taxes
            tax_calculation = await self.calculate_taxes(user_id, year)
            
            # Generate report
            report_filename = f"reports/tax_report_{user_id}_{year}.json"
            Path("reports").mkdir(exist_ok=True)
            
            with open(report_filename, 'w') as f:
                json.dump(asdict(tax_calculation), f, indent=2, default=str)
            
            self.logger.info(f"Tax report generated: {report_filename}")
            return report_filename
            
        except Exception as e:
            self.logger.error(f"Error generating tax report: {e}")
            raise
    
    async def _generate_risk_report(self, parameters: Dict[str, Any]) -> str:
        """Generate risk report."""
        try:
            date = parameters.get('date', datetime.now().date())
            
            # Calculate risk metrics (simplified)
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                var_95=0.05,  # 5% VaR
                var_99=0.08,  # 8% VaR
                expected_shortfall=0.10,
                max_drawdown=0.15,
                volatility=0.20,
                beta=1.2,
                sharpe_ratio=1.5,
                concentration_risk=0.25,
                liquidity_risk=0.10
            )
            
            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO risk_metrics 
                        (timestamp, var_95, var_99, expected_shortfall, max_drawdown,
                         volatility, beta, sharpe_ratio, concentration_risk, liquidity_risk)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, risk_metrics.timestamp, risk_metrics.var_95, risk_metrics.var_99,
                        risk_metrics.expected_shortfall, risk_metrics.max_drawdown,
                        risk_metrics.volatility, risk_metrics.beta, risk_metrics.sharpe_ratio,
                        risk_metrics.concentration_risk, risk_metrics.liquidity_risk)
            
            # Generate report
            report_filename = f"reports/risk_report_{date}.json"
            Path("reports").mkdir(exist_ok=True)
            
            with open(report_filename, 'w') as f:
                json.dump(asdict(risk_metrics), f, indent=2, default=str)
            
            self.logger.info(f"Risk report generated: {report_filename}")
            return report_filename
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            raise
    
    async def _generate_aml_report(self, parameters: Dict[str, Any]) -> str:
        """Generate AML report."""
        try:
            start_date = parameters.get('start_date', datetime.now().date() - timedelta(days=30))
            end_date = parameters.get('end_date', datetime.now().date())
            
            if not self.db_pool:
                raise ValueError("Database not available")
            
            async with self.db_pool.acquire() as conn:
                alerts = await conn.fetch("""
                    SELECT alert_id, timestamp, user_id, alert_type, severity,
                           description, investigation_status, reported_to_authorities
                    FROM aml_alerts 
                    WHERE DATE(timestamp) BETWEEN $1 AND $2
                    ORDER BY timestamp
                """, start_date, end_date)
                
                # Generate CSV report
                report_filename = f"reports/aml_report_{start_date}_{end_date}.csv"
                Path("reports").mkdir(exist_ok=True)
                
                with open(report_filename, 'w', newline='') as csvfile:
                    fieldnames = ['alert_id', 'timestamp', 'user_id', 'alert_type', 
                                'severity', 'description', 'investigation_status', 
                                'reported_to_authorities']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for alert in alerts:
                        writer.writerow(dict(alert))
                
                self.logger.info(f"AML report generated: {report_filename}")
                return report_filename
                
        except Exception as e:
            self.logger.error(f"Error generating AML report: {e}")
            raise
    
    async def _generate_audit_trail(self, parameters: Dict[str, Any]) -> str:
        """Generate audit trail report."""
        try:
            start_date = parameters.get('start_date', datetime.now().date() - timedelta(days=7))
            end_date = parameters.get('end_date', datetime.now().date())
            
            # Read audit log file
            audit_log_file = 'logs/audit_trail.log'
            report_filename = f"reports/audit_trail_{start_date}_{end_date}.txt"
            Path("reports").mkdir(exist_ok=True)
            
            with open(audit_log_file, 'r') as infile, open(report_filename, 'w') as outfile:
                for line in infile:
                    # Filter by date range (simplified)
                    if start_date.strftime('%Y-%m-%d') <= line[:10] <= end_date.strftime('%Y-%m-%d'):
                        outfile.write(line)
            
            self.logger.info(f"Audit trail generated: {report_filename}")
            return report_filename
            
        except Exception as e:
            self.logger.error(f"Error generating audit trail: {e}")
            raise
    
    async def _compliance_monitoring_loop(self):
        """Main compliance monitoring loop."""
        while True:
            try:
                # Check for compliance violations
                await self._monitor_compliance_violations()
                
                # Check for remediation deadlines
                await self._check_remediation_deadlines()
                
                # Generate scheduled reports
                await self._generate_scheduled_reports()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in compliance monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _aml_monitoring_loop(self):
        """AML monitoring loop."""
        while True:
            try:
                # Monitor for suspicious patterns
                await self._monitor_suspicious_patterns()
                
                # Update investigation status
                await self._update_investigation_status()
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in AML monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _report_generation_loop(self):
        """Automated report generation loop."""
        while True:
            try:
                current_time = datetime.now()
                
                # Generate daily reports at midnight
                if current_time.hour == 0 and current_time.minute < 10:
                    await self._generate_daily_reports()
                
                # Generate weekly reports on Monday
                if current_time.weekday() == 0 and current_time.hour == 1:
                    await self._generate_weekly_reports()
                
                # Generate monthly reports on first day of month
                if current_time.day == 1 and current_time.hour == 2:
                    await self._generate_monthly_reports()
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in report generation: {e}")
                await asyncio.sleep(300)
    
    async def _monitor_compliance_violations(self):
        """Monitor for compliance violations."""
        # Implementation would check for new violations
        pass
    
    async def _check_remediation_deadlines(self):
        """Check for approaching remediation deadlines."""
        try:
            current_time = datetime.now()
            warning_threshold = current_time + timedelta(days=7)
            
            for event in self.compliance_events:
                if (event.remediation_required and 
                    event.status == 'open' and 
                    event.remediation_deadline and 
                    event.remediation_deadline <= warning_threshold):
                    
                    self.logger.warning(f"Remediation deadline approaching for {event.event_id}: {event.remediation_deadline}")
                    
        except Exception as e:
            self.logger.error(f"Error checking remediation deadlines: {e}")
    
    async def _generate_scheduled_reports(self):
        """Generate scheduled compliance reports."""
        # Implementation would generate required reports
        pass
    
    async def _monitor_suspicious_patterns(self):
        """Monitor for suspicious trading patterns."""
        # Implementation would analyze trading patterns
        pass
    
    async def _update_investigation_status(self):
        """Update AML investigation status."""
        # Implementation would update investigation progress
        pass
    
    async def _generate_daily_reports(self):
        """Generate daily compliance reports."""
        try:
            yesterday = datetime.now().date() - timedelta(days=1)
            
            # Generate trade report
            await self.generate_report(ReportType.TRADE_REPORT, {
                'start_date': yesterday,
                'end_date': yesterday
            })
            
            # Generate risk report
            await self.generate_report(ReportType.RISK_REPORT, {
                'date': yesterday
            })
            
        except Exception as e:
            self.logger.error(f"Error generating daily reports: {e}")
    
    async def _generate_weekly_reports(self):
        """Generate weekly compliance reports."""
        try:
            end_date = datetime.now().date() - timedelta(days=1)
            start_date = end_date - timedelta(days=7)
            
            # Generate AML report
            await self.generate_report(ReportType.AML_REPORT, {
                'start_date': start_date,
                'end_date': end_date
            })
            
        except Exception as e:
            self.logger.error(f"Error generating weekly reports: {e}")
    
    async def _generate_monthly_reports(self):
        """Generate monthly compliance reports."""
        try:
            # Generate compliance summary
            await self.generate_report(ReportType.COMPLIANCE_SUMMARY, {
                'month': datetime.now().month,
                'year': datetime.now().year
            })
            
        except Exception as e:
            self.logger.error(f"Error generating monthly reports: {e}")
    
    async def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance status summary."""
        return {
            'compliance_events': len(self.compliance_events),
            'open_events': len([e for e in self.compliance_events if e.status == 'open']),
            'critical_events': len([e for e in self.compliance_events if e.severity == ComplianceLevel.CRITICAL]),
            'aml_alerts': len(self.aml_alerts),
            'pending_investigations': len([a for a in self.aml_alerts if a.investigation_status == 'pending']),
            'pending_reports': len(self.pending_reports),
            'last_updated': datetime.now().isoformat()
        }
    
    async def close(self):
        """Close compliance reporting connections."""
        try:
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("Compliance reporting system closed")
        except Exception as e:
            self.logger.error(f"Error closing compliance system: {e}")


async def main():
    """Main compliance entry point."""
    from production_config import get_production_config
    
    config = get_production_config()
    compliance = ComplianceReporting(config)
    
    try:
        await compliance.initialize()
        
        # Example: Record a trade
        trade_data = {
            'trade_id': 'TEST_001',
            'timestamp': datetime.now(),
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 1.0,
            'price': 50000.0,
            'exchange': 'binance',
            'user_id': 'user_123'
        }
        
        await compliance.record_trade(trade_data)
        
        # Example: Generate trade report
        report_file = await compliance.generate_report(ReportType.TRADE_REPORT, {
            'start_date': datetime.now().date() - timedelta(days=7),
            'end_date': datetime.now().date()
        })
        
        print(f"Report generated: {report_file}")
        
        # Keep running
        await asyncio.sleep(3600)
        
    except KeyboardInterrupt:
        print("\nShutting down compliance system...")
    finally:
        await compliance.close()


if __name__ == "__main__":
    asyncio.run(main())