"""
Live Dashboard - Real-time Web Interface
Dash-based dashboard for monitoring and controlling the trading system
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import requests
import json

from config import get_config

logger = logging.getLogger(__name__)

class TradingDashboard:
    """
    Real-time trading dashboard using Dash
    """
    
    def __init__(self):
        self.config = get_config()
        self.api_base_url = f"http://{self.config.API_HOST}:{self.config.API_PORT}"
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True
        )
        
        self.app.title = "OmniBeing Trading Dashboard"
        
        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
        
        logger.info("Trading Dashboard initialized")
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        
        # Header
        header = dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-chart-line me-3"),
                    "OmniBeing Trading Dashboard"
                ], className="text-primary mb-0"),
                html.P("Advanced AI Trading System", className="text-muted")
            ], width=8),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-play me-2"),
                    "Start System"
                ], id="btn-start", color="success", className="me-2"),
                dbc.Button([
                    html.I(className="fas fa-stop me-2"),
                    "Stop System"
                ], id="btn-stop", color="danger", className="me-2"),
                dbc.Button([
                    html.I(className="fas fa-sync-alt"),
                ], id="btn-refresh", color="primary")
            ], width=4, className="text-end")
        ], className="mb-4")
        
        # Status cards
        status_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="portfolio-value", children="$0.00", className="text-success"),
                        html.P("Portfolio Value", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="daily-pnl", children="$0.00", className="text-info"),
                        html.P("Daily P&L", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="open-positions", children="0", className="text-warning"),
                        html.P("Open Positions", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="risk-level", children="Low", className="text-primary"),
                        html.P("Risk Level", className="card-text")
                    ])
                ])
            ], width=3)
        ], className="mb-4")
        
        # Main content tabs
        tabs = dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="tab-overview"),
            dbc.Tab(label="Trading", tab_id="tab-trading"),
            dbc.Tab(label="Risk Management", tab_id="tab-risk"),
            dbc.Tab(label="Performance", tab_id="tab-performance"),
            dbc.Tab(label="Logs", tab_id="tab-logs"),
            dbc.Tab(label="Backtesting", tab_id="tab-backtest")
        ], id="main-tabs", active_tab="tab-overview")
        
        # Tab content
        tab_content = html.Div(id="tab-content", className="mt-3")
        
        # Alerts container
        alerts = html.Div(id="alerts-container")
        
        # Auto-refresh interval
        interval = dcc.Interval(
            id='interval-component',
            interval=5000,  # Update every 5 seconds
            n_intervals=0
        )
        
        # Complete layout
        self.app.layout = dbc.Container([
            dcc.Store(id='system-data'),
            header,
            alerts,
            status_cards,
            tabs,
            tab_content,
            interval
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('system-data', 'data'),
             Output('portfolio-value', 'children'),
             Output('daily-pnl', 'children'),
             Output('open-positions', 'children'),
             Output('risk-level', 'children'),
             Output('alerts-container', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('btn-refresh', 'n_clicks')],
            prevent_initial_call=False
        )
        def update_system_data(n_intervals, refresh_clicks):
            """Update system data and status cards"""
            try:
                # Fetch system status
                response = requests.get(f"{self.api_base_url}/system/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract key metrics
                    portfolio_value = data.get('risk_manager', {}).get('portfolio_value', 0)
                    total_pnl = data.get('trading_system', {}).get('performance', {}).get('total_pnl', 0)
                    open_positions = data.get('trading_system', {}).get('active_positions', 0)
                    risk_level = data.get('risk_manager', {}).get('risk_metrics', {}).get('risk_level', 'unknown')
                    
                    # Format values
                    portfolio_str = f"${portfolio_value:,.2f}"
                    pnl_str = f"${total_pnl:,.2f}"
                    positions_str = str(open_positions)
                    risk_str = risk_level.title()
                    
                    # Check for alerts
                    alerts = []
                    if risk_level in ['high', 'critical']:
                        alerts.append(
                            dbc.Alert(
                                f"High risk level detected: {risk_level}",
                                color="warning",
                                dismissable=True
                            )
                        )
                    
                    return data, portfolio_str, pnl_str, positions_str, risk_str, alerts
                
                else:
                    # API not available
                    return {}, "$0.00", "$0.00", "0", "Unknown", [
                        dbc.Alert("API connection failed", color="danger", dismissable=True)
                    ]
            
            except Exception as e:
                logger.error(f"Error updating system data: {e}")
                return {}, "$0.00", "$0.00", "0", "Error", [
                    dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)
                ]
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('system-data', 'data')]
        )
        def update_tab_content(active_tab, system_data):
            """Update tab content based on selected tab"""
            if active_tab == "tab-overview":
                return self.create_overview_tab(system_data)
            elif active_tab == "tab-trading":
                return self.create_trading_tab(system_data)
            elif active_tab == "tab-risk":
                return self.create_risk_tab(system_data)
            elif active_tab == "tab-performance":
                return self.create_performance_tab(system_data)
            elif active_tab == "tab-logs":
                return self.create_logs_tab(system_data)
            elif active_tab == "tab-backtest":
                return self.create_backtest_tab(system_data)
            else:
                return html.Div("Select a tab")
        
        @self.app.callback(
            Output('alerts-container', 'children', allow_duplicate=True),
            [Input('btn-start', 'n_clicks'),
             Input('btn-stop', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_system_controls(start_clicks, stop_clicks):
            """Handle start/stop system buttons"""
            ctx = callback_context
            if not ctx.triggered:
                return []
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'btn-start':
                    response = requests.post(f"{self.api_base_url}/system/start")
                    if response.status_code == 200:
                        return [dbc.Alert("System started successfully", color="success", dismissable=True)]
                    else:
                        return [dbc.Alert("Failed to start system", color="danger", dismissable=True)]
                
                elif button_id == 'btn-stop':
                    response = requests.post(f"{self.api_base_url}/system/stop")
                    if response.status_code == 200:
                        return [dbc.Alert("System stopped successfully", color="info", dismissable=True)]
                    else:
                        return [dbc.Alert("Failed to stop system", color="danger", dismissable=True)]
            
            except Exception as e:
                return [dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)]
            
            return []
    
    def create_overview_tab(self, system_data: Dict[str, Any]) -> html.Div:
        """Create overview tab content"""
        
        # Price chart
        price_chart = self.create_price_chart()
        
        # System status
        status_table = self.create_status_table(system_data)
        
        # Recent trades
        trades_table = self.create_recent_trades_table()
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Price Charts"),
                    dbc.CardBody([price_chart])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Status"),
                    dbc.CardBody([status_table])
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Recent Trades"),
                    dbc.CardBody([trades_table])
                ])
            ], width=4)
        ])
    
    def create_trading_tab(self, system_data: Dict[str, Any]) -> html.Div:
        """Create trading tab content"""
        
        # Manual trading form
        trading_form = dbc.Card([
            dbc.CardHeader("Manual Trading"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Symbol"),
                        dbc.Select(
                            id="trade-symbol",
                            options=[
                                {"label": "BTC/USDT", "value": "BTCUSDT"},
                                {"label": "ETH/USDT", "value": "ETHUSDT"},
                                {"label": "ADA/USDT", "value": "ADAUSDT"}
                            ],
                            value="BTCUSDT"
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Side"),
                        dbc.Select(
                            id="trade-side",
                            options=[
                                {"label": "Buy", "value": "buy"},
                                {"label": "Sell", "value": "sell"}
                            ],
                            value="buy"
                        )
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Amount"),
                        dbc.Input(id="trade-amount", type="number", value=0.01, step=0.001)
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Type"),
                        dbc.Select(
                            id="trade-type",
                            options=[
                                {"label": "Market", "value": "market"},
                                {"label": "Limit", "value": "limit"}
                            ],
                            value="market"
                        )
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Price (for limit)"),
                        dbc.Input(id="trade-price", type="number", step=0.01)
                    ], width=2),
                    dbc.Col([
                        dbc.Label(" "),
                        dbc.Button("Place Order", id="btn-place-order", color="primary")
                    ], width=1)
                ])
            ])
        ])
        
        # Positions table
        positions_table = self.create_positions_table(system_data)
        
        # Order book (placeholder)
        order_book = dbc.Card([
            dbc.CardHeader("Order Book"),
            dbc.CardBody([
                html.P("Order book data would be displayed here")
            ])
        ])
        
        return html.Div([
            trading_form,
            html.Br(),
            dbc.Row([
                dbc.Col([positions_table], width=8),
                dbc.Col([order_book], width=4)
            ])
        ])
    
    def create_risk_tab(self, system_data: Dict[str, Any]) -> html.Div:
        """Create risk management tab content"""
        
        # Risk metrics
        risk_metrics = self.create_risk_metrics_chart(system_data)
        
        # Risk controls
        risk_controls = dbc.Card([
            dbc.CardHeader("Risk Controls"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Risk Level"),
                        dbc.Input(id="risk-level-input", type="number", value=0.02, step=0.01, min=0, max=1)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Max Position Size"),
                        dbc.Input(id="max-position-input", type="number", value=0.1, step=0.01, min=0, max=1)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Stop Loss %"),
                        dbc.Input(id="stop-loss-input", type="number", value=0.05, step=0.01, min=0, max=1)
                    ], width=3),
                    dbc.Col([
                        dbc.Button("Update", id="btn-update-risk", color="primary"),
                        html.Br(),
                        dbc.Button("Emergency Stop", id="btn-emergency-stop", color="danger", className="mt-2")
                    ], width=3)
                ])
            ])
        ])
        
        # Risk recommendations
        recommendations = self.create_risk_recommendations(system_data)
        
        return html.Div([
            risk_controls,
            html.Br(),
            dbc.Row([
                dbc.Col([risk_metrics], width=8),
                dbc.Col([recommendations], width=4)
            ])
        ])
    
    def create_performance_tab(self, system_data: Dict[str, Any]) -> html.Div:
        """Create performance tab content"""
        
        # Equity curve
        equity_chart = self.create_equity_curve()
        
        # Performance metrics
        performance_metrics = self.create_performance_metrics(system_data)
        
        # Trade distribution
        trade_distribution = self.create_trade_distribution()
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Equity Curve"),
                    dbc.CardBody([equity_chart])
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("Trade Distribution"),
                    dbc.CardBody([trade_distribution])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Metrics"),
                    dbc.CardBody([performance_metrics])
                ])
            ], width=4)
        ])
    
    def create_logs_tab(self, system_data: Dict[str, Any]) -> html.Div:
        """Create logs tab content"""
        
        # Log filters
        log_filters = dbc.Card([
            dbc.CardHeader("Log Filters"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Log Type"),
                        dbc.Select(
                            id="log-type-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Trades", "value": "trades"},
                                {"label": "Decisions", "value": "decisions"},
                                {"label": "Performance", "value": "performance"}
                            ],
                            value="all"
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Limit"),
                        dbc.Input(id="log-limit", type="number", value=100, min=1, max=1000)
                    ], width=2),
                    dbc.Col([
                        dbc.Button("Refresh Logs", id="btn-refresh-logs", color="primary")
                    ], width=2)
                ])
            ])
        ])
        
        # Logs table
        logs_table = html.Div(id="logs-table-container")
        
        return html.Div([
            log_filters,
            html.Br(),
            logs_table
        ])
    
    def create_backtest_tab(self, system_data: Dict[str, Any]) -> html.Div:
        """Create backtesting tab content"""
        
        # Backtest form
        backtest_form = dbc.Card([
            dbc.CardHeader("Run Backtest"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Strategy"),
                        dbc.Select(
                            id="backtest-strategy",
                            options=[
                                {"label": "Moving Average Cross", "value": "ma_cross"},
                                {"label": "RSI", "value": "rsi"}
                            ],
                            value="ma_cross"
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Start Date"),
                        dbc.Input(id="backtest-start", type="date", value="2023-01-01")
                    ], width=3),
                    dbc.Col([
                        dbc.Label("End Date"),
                        dbc.Input(id="backtest-end", type="date", value="2023-12-31")
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Initial Balance"),
                        dbc.Input(id="backtest-balance", type="number", value=100000)
                    ], width=3)
                ]),
                html.Br(),
                dbc.Button("Run Backtest", id="btn-run-backtest", color="primary")
            ])
        ])
        
        # Backtest results
        backtest_results = html.Div(id="backtest-results-container")
        
        return html.Div([
            backtest_form,
            html.Br(),
            backtest_results
        ])
    
    def create_price_chart(self) -> dcc.Graph:
        """Create price chart"""
        try:
            # Fetch price data
            response = requests.get(f"{self.api_base_url}/data/prices", timeout=5)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', {})
                
                # Create chart
                fig = go.Figure()
                
                for symbol, price_data in prices.items():
                    if isinstance(price_data, dict) and 'price' in price_data:
                        fig.add_trace(go.Scatter(
                            x=[datetime.now()],
                            y=[price_data['price']],
                            mode='markers',
                            name=symbol,
                            marker=dict(size=10)
                        ))
                
                fig.update_layout(
                    title="Current Prices",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=400
                )
                
                return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
        
        # Return empty chart on error
        fig = go.Figure()
        fig.update_layout(title="Price data not available", height=400)
        return dcc.Graph(figure=fig)
    
    def create_status_table(self, system_data: Dict[str, Any]) -> dbc.Table:
        """Create system status table"""
        if not system_data:
            return dbc.Table([
                html.Thead([html.Tr([html.Th("Component"), html.Th("Status")])]),
                html.Tbody([html.Tr([html.Td("No data"), html.Td("Available")])])
            ])
        
        status_items = []
        trading_status = system_data.get('trading_system', {})
        
        # Add status items
        status_items.append(["Trading System", "Running" if trading_status.get('running') else "Stopped"])
        status_items.append(["Total Trades", str(trading_status.get('performance', {}).get('total_trades', 0))])
        status_items.append(["Active Positions", str(trading_status.get('active_positions', 0))])
        
        return dbc.Table([
            html.Thead([html.Tr([html.Th("Component"), html.Th("Status")])]),
            html.Tbody([
                html.Tr([html.Td(item[0]), html.Td(item[1])]) 
                for item in status_items
            ])
        ], striped=True, bordered=True, hover=True, size="sm")
    
    def create_recent_trades_table(self) -> dbc.Table:
        """Create recent trades table"""
        try:
            response = requests.get(f"{self.api_base_url}/logs/trades?limit=5", timeout=5)
            if response.status_code == 200:
                data = response.json()
                trades = data.get('trades', [])
                
                if trades:
                    return dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Time"),
                                html.Th("Symbol"),
                                html.Th("Side"),
                                html.Th("Amount"),
                                html.Th("Status")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(trade.get('timestamp', '')[:19]),
                                html.Td(trade.get('symbol', '')),
                                html.Td(trade.get('action', '')),
                                html.Td(f"{trade.get('quantity', 0):.4f}"),
                                html.Td(trade.get('status', ''))
                            ]) for trade in trades[:5]
                        ])
                    ], striped=True, bordered=True, hover=True, size="sm")
        
        except Exception as e:
            logger.error(f"Error creating trades table: {e}")
        
        return dbc.Table([
            html.Thead([html.Tr([html.Th("No trades available")])]),
            html.Tbody([])
        ])
    
    def create_positions_table(self, system_data: Dict[str, Any]) -> dbc.Card:
        """Create positions table"""
        # This would fetch and display current positions
        return dbc.Card([
            dbc.CardHeader("Current Positions"),
            dbc.CardBody([
                html.P("No open positions")
            ])
        ])
    
    def create_risk_metrics_chart(self, system_data: Dict[str, Any]) -> dcc.Graph:
        """Create risk metrics chart"""
        # Create a simple risk gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 0.3,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        fig.update_layout(height=300)
        return dcc.Graph(figure=fig)
    
    def create_risk_recommendations(self, system_data: Dict[str, Any]) -> dbc.Card:
        """Create risk recommendations card"""
        return dbc.Card([
            dbc.CardHeader("Risk Recommendations"),
            dbc.CardBody([
                html.P("• Maintain current position sizes"),
                html.P("• Monitor BTC correlation"),
                html.P("• Consider reducing exposure if volatility increases")
            ])
        ])
    
    def create_equity_curve(self) -> dcc.Graph:
        """Create equity curve chart"""
        # Placeholder equity curve
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        equity = 100000 + (dates - dates[0]).days * 50 + np.random.normal(0, 1000, len(dates)).cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Equity'))
        fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
        
        return dcc.Graph(figure=fig)
    
    def create_performance_metrics(self, system_data: Dict[str, Any]) -> dbc.Table:
        """Create performance metrics table"""
        metrics = [
            ["Total Return", "15.2%"],
            ["Sharpe Ratio", "1.45"],
            ["Max Drawdown", "8.3%"],
            ["Win Rate", "62%"],
            ["Profit Factor", "1.8"]
        ]
        
        return dbc.Table([
            html.Tbody([
                html.Tr([html.Td(metric[0]), html.Td(metric[1])]) 
                for metric in metrics
            ])
        ], striped=True, bordered=True, hover=True)
    
    def create_trade_distribution(self) -> dcc.Graph:
        """Create trade distribution chart"""
        # Placeholder data
        returns = np.random.normal(0.02, 0.15, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=20, name='Trade Returns'))
        fig.update_layout(title="Trade Return Distribution", xaxis_title="Return", yaxis_title="Frequency")
        
        return dcc.Graph(figure=fig)
    
    def run(self):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on {self.config.DASHBOARD_HOST}:{self.config.DASHBOARD_PORT}")
        
        self.app.run_server(
            host=self.config.DASHBOARD_HOST,
            port=self.config.DASHBOARD_PORT,
            debug=self.config.DASHBOARD_DEBUG
        )

# Create global dashboard instance
dashboard = None

def get_dashboard() -> TradingDashboard:
    """Get the global dashboard instance"""
    global dashboard
    if dashboard is None:
        dashboard = TradingDashboard()
    return dashboard

if __name__ == "__main__":
    # Run the dashboard
    dashboard = get_dashboard()
    dashboard.run()