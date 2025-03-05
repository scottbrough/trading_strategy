"""
Real-time monitoring dashboard for the trading system.
Provides performance metrics, trade visualization, and system health monitoring.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from typing import Dict, List, Optional

##these imports are what are causing the dashboard to not show correctly. 
from src.core.config import config
from src.core.logger import log_manager
from src.data.database import db

logger = log_manager.get_logger(__name__)

class DashboardManager:
    def __init__(self):
        """Initialize dashboard manager"""
        self.config = config.get_monitoring_config()
        self.update_interval = self.config['update_interval']
        self.metrics_queue = queue.Queue()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="Trading System Dashboard"
        )
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col(html.H1("Trading System Dashboard", 
                               className="text-center text-primary mb-4"))
            ]),
            
            # Performance Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics"),
                        dbc.CardBody([
                            html.Div(id='performance-metrics'),
                            dcc.Interval(
                                id='metrics-update',
                                interval=5*1000  # 5 seconds
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                # Equity Curve
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Equity Curve"),
                        dbc.CardBody([
                            dcc.Graph(id='equity-chart'),
                            dcc.Interval(
                                id='equity-update',
                                interval=30*1000  # 30 seconds
                            )
                        ])
                    ])
                ], width=6),
                
                # Trade Distribution
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Trade Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id='trade-distribution'),
                            dcc.Interval(
                                id='distribution-update',
                                interval=60*1000  # 1 minute
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Strategy Performance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Performance"),
                        dbc.CardBody([
                            html.Div(id='strategy-metrics'),
                            dbc.Table(id='strategy-table'),
                            dcc.Interval(
                                id='strategy-update',
                                interval=10*1000  # 10 seconds
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Risk Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody([
                            html.Div(id='risk-metrics'),
                            dcc.Interval(
                                id='risk-update',
                                interval=15*1000  # 15 seconds
                            )
                        ])
                    ])
                ], width=6),
                
                # System Health
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Health"),
                        dbc.CardBody([
                            html.Div(id='system-health'),
                            dcc.Interval(
                                id='health-update',
                                interval=5*1000  # 5 seconds
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Active Trades
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Active Trades"),
                        dbc.CardBody([
                            html.Div(id='active-trades'),
                            dcc.Interval(
                                id='trades-update',
                                interval=5*1000  # 5 seconds
                            )
                        ])
                    ])
                ])
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('performance-metrics', 'children'),
            Input('metrics-update', 'n_intervals')
        )
        def update_performance_metrics(n):
            metrics = self._get_performance_metrics()
            return self._create_metrics_display(metrics)
        
        @self.app.callback(
            Output('equity-chart', 'figure'),
            Input('equity-update', 'n_intervals')
        )
        def update_equity_chart(n):
            return self._create_equity_chart()
        
        @self.app.callback(
            Output('trade-distribution', 'figure'),
            Input('distribution-update', 'n_intervals')
        )
        def update_trade_distribution(n):
            return self._create_trade_distribution()
        
        @self.app.callback(
            Output('strategy-table', 'children'),
            Input('strategy-update', 'n_intervals')
        )
        def update_strategy_metrics(n):
            return self._create_strategy_table()
        
        @self.app.callback(
            Output('risk-metrics', 'children'),
            Input('risk-update', 'n_intervals')
        )
        def update_risk_metrics(n):
            metrics = self._get_risk_metrics()
            return self._create_risk_display(metrics)
        
        @self.app.callback(
            Output('system-health', 'children'),
            Input('health-update', 'n_intervals')
        )
        def update_system_health(n):
            health = self._get_system_health()
            return self._create_health_display(health)
        
        @self.app.callback(
            Output('active-trades', 'children'),
            Input('trades-update', 'n_intervals')
        )
        def update_active_trades(n):
            trades = self._get_active_trades()
            return self._create_trades_table(trades)
    
    def _get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            trades_df = db.get_trades()
            if trades_df.empty:
                return {}
            
            # Calculate metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = trades_df['pnl'].sum()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean())
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': avg_win / avg_loss if avg_loss != 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    def _create_metrics_display(self, metrics: Dict) -> html.Div:
        """Create performance metrics display"""
        try:
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4(f"Total Trades: {metrics.get('total_trades', 0)}"),
                        html.H4(f"Win Rate: {metrics.get('win_rate', 0):.2%}"),
                        html.H4(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}"),
                        html.H4(f"Avg Win: ${metrics.get('avg_win', 0):,.2f}"),
                        html.H4(f"Avg Loss: ${metrics.get('avg_loss', 0):,.2f}"),
                        html.H4(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                    ])
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error creating metrics display: {str(e)}")
            return html.Div("Error loading metrics")
    
    def _create_equity_chart(self) -> dict:
        """Create equity curve chart"""
        try:
            # Get trade history
            trades_df = db.get_trades()
            if trades_df.empty:
                return {}
            
            # Calculate cumulative equity
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            figure = {
                'data': [
                    go.Scatter(
                        x=trades_df.index,
                        y=trades_df['cumulative_pnl'],
                        mode='lines',
                        name='Equity'
                    )
                ],
                'layout': go.Layout(
                    title='Equity Curve',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Equity ($)'},
                    template='plotly_dark'
                )
            }
            
            return figure
            
        except Exception as e:
            logger.error(f"Error creating equity chart: {str(e)}")
            return {}
    
    def _create_trade_distribution(self) -> dict:
        """Create trade P&L distribution chart"""
        try:
            trades_df = db.get_trades()
            if trades_df.empty:
                return {}
            
            figure = {
                'data': [
                    go.Histogram(
                        x=trades_df['pnl'],
                        nbinsx=50,
                        name='P&L Distribution'
                    )
                ],
                'layout': go.Layout(
                    title='Trade P&L Distribution',
                    xaxis={'title': 'P&L ($)'},
                    yaxis={'title': 'Frequency'},
                    template='plotly_dark'
                )
            }
            
            return figure
            
        except Exception as e:
            logger.error(f"Error creating trade distribution: {str(e)}")
            return {}
    
    def _create_strategy_table(self) -> html.Table:
        """Create strategy performance table"""
        try:
            trades_df = db.get_trades()
            if trades_df.empty:
                return html.Table()
            
            # Group by strategy
            strategy_stats = trades_df.groupby('strategy').agg({
                'pnl': ['count', 'sum', 'mean'],
                'win_rate': 'mean',
                'sharpe_ratio': 'mean'
            }).round(2)
            
            # Create table
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Strategy"),
                    html.Th("# Trades"),
                    html.Th("Total P&L"),
                    html.Th("Avg P&L"),
                    html.Th("Win Rate"),
                    html.Th("Sharpe")
                ]))
            ]
            
            rows = []
            for strategy in strategy_stats.index:
                stats = strategy_stats.loc[strategy]
                row = html.Tr([
                    html.Td(strategy),
                    html.Td(f"{stats[('pnl', 'count')]}"),
                    html.Td(f"${stats[('pnl', 'sum')]:,.2f}"),
                    html.Td(f"${stats[('pnl', 'mean')]:,.2f}"),
                    html.Td(f"{stats[('win_rate', 'mean')]:.2%}"),
                    html.Td(f"{stats[('sharpe_ratio', 'mean')]:.2f}")
                ])
                rows.append(row)
            
            table_body = [html.Tbody(rows)]
            return dbc.Table(
                table_header + table_body,
                bordered=True,
                dark=True,
                hover=True,
                responsive=True,
                striped=True
            )
            
        except Exception as e:
            logger.error(f"Error creating strategy table: {str(e)}")
            return html.Table()
    
    def _get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        try:
            trades_df = db.get_trades(status='open')
            
            metrics = {
                'exposure': sum(trades_df['amount'] * trades_df['entry_price']),
                'max_drawdown': self._calculate_drawdown(),
                'var_95': self._calculate_var(),
                'positions': len(trades_df),
                'largest_position': max(trades_df['amount'] * trades_df['entry_price']) if not trades_df.empty else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {str(e)}")
            return {}
    
    def _create_risk_display(self, metrics: Dict) -> html.Div:
        """Create risk metrics display"""
        try:
            # Create risk indicators with color coding
            exposure_color = 'success' if metrics.get('exposure', 0) < self.config.get('risk_limits', {}).get('max_exposure', 1000000) else 'danger'
            drawdown_color = 'success' if metrics.get('max_drawdown', 0) < self.config.get('risk_limits', {}).get('max_drawdown', 0.2) else 'danger'
            
            return html.Div([
                dbc.Alert(
                    f"Current Exposure: ${metrics.get('exposure', 0):,.2f}",
                    color=exposure_color,
                    className="mb-2"
                ),
                dbc.Alert(
                    f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
                    color=drawdown_color,
                    className="mb-2"
                ),
                html.H5(f"VaR (95%): ${metrics.get('var_95', 0):,.2f}"),
                html.H5(f"Open Positions: {metrics.get('positions', 0)}"),
                html.H5(f"Largest Position: ${metrics.get('largest_position', 0):,.2f}")
            ])
            
        except Exception as e:
            logger.error(f"Error creating risk display: {str(e)}")
            return html.Div("Error loading risk metrics")
    
    def _calculate_drawdown(self):
        """Placeholder for drawdown calculation"""
        return 0.0
        
    def _calculate_var(self):
        """Placeholder for VaR calculation"""
        return 0.0
    
    def _get_active_trades(self) -> pd.DataFrame:
        """Get current active trades"""
        try:
            return db.get_trades(status='open')
        except Exception as e:
            logger.error(f"Error getting active trades: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_unrealized_pnl(self, trade):
        """Calculate unrealized P&L for a position"""
        try:
            # Get current price
            symbol = trade['symbol']
            entry_price = trade['entry_price']
            size = trade['size']
            side = trade['side']
            
            # Use latest price from paper trading executor if available
            latest_price = None
            try:
                from src.trading.paper_trading import paper_trading_executor
                if hasattr(paper_trading_executor, 'latest_prices') and symbol in paper_trading_executor.latest_prices:
                    latest_price = paper_trading_executor.latest_prices[symbol]
            except:
                pass
            
            # If not available, try to get from database
            if not latest_price:
                try:
                    from src.data.database import db
                    recent_ohlcv = db.get_ohlcv(symbol, '1m', 
                                            datetime.now() - timedelta(minutes=10),
                                            datetime.now())
                    if not recent_ohlcv.empty:
                        latest_price = recent_ohlcv['close'].iloc[-1]
                except:
                    pass
                    
            # If still no price, use entry price
            if not latest_price:
                latest_price = entry_price
                
            # Calculate unrealized P&L
            if side == 'long':
                return (latest_price - entry_price) * size
            else:  # short
                return (entry_price - latest_price) * size
                
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {str(e)}")
            return 0.0

    def _create_metrics_display(self, metrics: Dict) -> html.Div:
        """Create performance metrics display with account balance"""
        try:
            # Get current account balance
            balance = 0
            equity = 0
            unrealized_pnl = 0
            
            # Try to get balance from paper trading executor
            try:
                # Import here to avoid circular imports
                import importlib
                paper_trading = importlib.import_module('src.trading.paper_trading')
                if hasattr(paper_trading, 'paper_trading_executor'):
                    account_info = paper_trading.paper_trading_executor.get_account_balance()
                    balance = account_info.get('balance', 0)
                    equity = account_info.get('equity', 0)
                    unrealized_pnl = account_info.get('unrealized_pnl', 0)
            except Exception as e:
                logger.warning(f"Could not get account balance from executor: {str(e)}")
            
            # If we couldn't get from executor, use metrics
            if balance == 0:
                balance = metrics.get('current_balance', self.config.get('initial_balance', 10000))
                equity = metrics.get('current_equity', balance)
                unrealized_pnl = equity - balance
            
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        # Account Balance Section
                        html.Div([
                            html.H3("Account Overview", className="text-primary"),
                            html.H4(f"Cash Balance: ${balance:,.2f}", className="mt-2"),
                            html.H4(f"Total Equity: ${equity:,.2f}", className="mt-2"),
                            html.H4(f"Unrealized P&L: ${unrealized_pnl:,.2f}", 
                                className=f"{'text-success' if unrealized_pnl >= 0 else 'text-danger'} mt-2")
                        ], className="p-3 border rounded bg-light mb-3"),
                        
                        # Performance Metrics Section
                        html.Div([
                            html.H3("Trading Performance", className="text-primary"),
                            html.H4(f"Total Trades: {metrics.get('total_trades', 0)}"),
                            html.H4(f"Win Rate: {metrics.get('win_rate', 0):.2%}"),
                            html.H4(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}"),
                            html.H4(f"Avg Win: ${metrics.get('avg_win', 0):,.2f}"),
                            html.H4(f"Avg Loss: ${metrics.get('avg_loss', 0):,.2f}"),
                            html.H4(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                        ], className="p-3 border rounded bg-light")
                    ])
                ])
            ])
                
        except Exception as e:
            logger.error(f"Error creating metrics display: {str(e)}")
            return html.Div("Error loading metrics")

    def _create_trades_table(self, trades_df: pd.DataFrame) -> html.Div:
        """Create active trades table with live prices"""
        try:
            if trades_df.empty:
                return html.Div("No active trades")
            
            # Try to get latest prices from paper trading executor
            latest_prices = {}
            try:
                # Import here to avoid circular imports
                import importlib
                paper_trading = importlib.import_module('src.trading.paper_trading')
                if hasattr(paper_trading, 'paper_trading_executor'):
                    latest_prices = paper_trading.paper_trading_executor.latest_prices
            except Exception as e:
                logger.warning(f"Could not get latest prices: {str(e)}")
            
            # Add current price based on latest data
            trades_df['current_price'] = trades_df.apply(
                lambda row: latest_prices.get(row['symbol'], row['entry_price']),
                axis=1
            )
            
            # Calculate unrealized P&L
            trades_df['unrealized_pnl'] = trades_df.apply(self._calculate_unrealized_pnl, axis=1)
            
            # Create table
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Side"),
                    html.Th("Entry Price"),
                    html.Th("Current Price"),
                    html.Th("Size"),
                    html.Th("Unrealized P&L"),
                    html.Th("P&L %"),
                    html.Th("Duration")
                ]))
            ]
            
            rows = []
            for _, trade in trades_df.iterrows():
                # Calculate P&L percentage
                entry_price = trade['entry_price']
                current_price = trade['current_price']
                pnl_pct = 0
                
                if trade['side'] == 'long':
                    pnl_pct = (current_price / entry_price - 1) * 100
                else:  # short
                    pnl_pct = (entry_price / current_price - 1) * 100
                
                # Determine row color based on P&L
                row_class = "table-success" if trade['unrealized_pnl'] > 0 else "table-danger"
                
                rows.append(html.Tr([
                    html.Td(trade['symbol']),
                    html.Td(trade['side']),
                    html.Td(f"${trade['entry_price']:,.2f}"),
                    html.Td(f"${trade['current_price']:,.2f}"),
                    html.Td(f"{trade['size']:.4f}"),
                    html.Td(f"${trade['unrealized_pnl']:,.2f}"),
                    html.Td(f"{pnl_pct:.2f}%"),
                    html.Td(self._format_duration(trade['entry_time']))
                ], className=row_class))
            
            table_body = [html.Tbody(rows)]
            
            return dbc.Table(
                table_header + table_body,
                bordered=True,
                dark=True,
                hover=True,
                responsive=True,
                striped=True
            )
            
        except Exception as e:
            logger.error(f"Error creating trades table: {str(e)}")
            return html.Div("Error loading trades")
    
    def _format_duration(self, timestamp):
        """Format time duration"""
        delta = datetime.now() - timestamp
        hours = delta.total_seconds() / 3600
        return f"{hours:.1f} hours"
    
    def _create_trades_table(self, trades_df: pd.DataFrame) -> html.Div:
        """Create active trades table"""
        try:
            if trades_df.empty:
                return html.Div("No active trades")
            
            # Add current price (placeholder)
            trades_df['current_price'] = trades_df['entry_price']
            
            # Calculate unrealized P&L
            trades_df['unrealized_pnl'] = trades_df.apply(self._calculate_unrealized_pnl, axis=1)
            
            # Create table
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Side"),
                    html.Th("Entry Price"),
                    html.Th("Current Price"),
                    html.Th("Size"),
                    html.Th("Unrealized P&L"),
                    html.Th("Duration")
                ]))
            ]
            
            rows = []
            for _, trade in trades_df.iterrows():
                rows.append(html.Tr([
                    html.Td(trade['symbol']),
                    html.Td(trade['side']),
                    html.Td(f"${trade['entry_price']:,.2f}"),
                    html.Td(f"${trade['current_price']:,.2f}"),
                    html.Td(f"{trade['amount']:.4f}"),
                    html.Td(f"${trade['unrealized_pnl']:,.2f}"),
                    html.Td(self._format_duration(trade['entry_time']))
                ]))
            
            table_body = [html.Tbody(rows)]
            
            return dbc.Table(
                table_header + table_body,
                bordered=True,
                dark=True,
                hover=True,
                responsive=True,
                striped=True
            )
            
        except Exception as e:
            logger.error(f"Error creating trades table: {str(e)}")
            return html.Div("Error loading trades")
    
    def _check_system_status(self):
        """Placeholder for system status check"""
        return "healthy"
    
    def _get_error_rate(self):
        """Placeholder for error rate calculation"""
        return "Low"
    
    def _get_memory_usage(self):
        """Placeholder for memory usage"""
        return "256MB"
    
    def _get_api_latency(self):
        """Placeholder for API latency"""
        return "120ms"
    
    def _get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            return {
                'status': self._check_system_status(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error_rate': self._get_error_rate(),
                'memory_usage': self._get_memory_usage(),
                'api_latency': self._get_api_latency()
            }
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {}
    
    def _create_health_display(self, health: Dict) -> html.Div:
        """Create health display"""
        status_color = 'success' if health.get('status') == 'healthy' else 'warning'
        
        return html.Div([
            dbc.Alert(
                f"Status: {health.get('status', 'unknown')}",
                color=status_color,
                className="mb-2"
            ),
            html.H5(f"Last Update: {health.get('last_update', 'unknown')}"),
            html.H5(f"Error Rate: {health.get('error_rate', 'unknown')}"),
            html.H5(f"Memory Usage: {health.get('memory_usage', 'unknown')}"),
            html.H5(f"API Latency: {health.get('api_latency', 'unknown')}")
        ])
    
    # Find the run_server method in DashboardManager class
    def run(self, host: str = '0.0.0.0', port: int = None):
        """Run the dashboard server"""
        try:
            port = port or self.config.get('dashboard_port', 8050)
            logger.info(f"Starting dashboard on {host}:{port}")
            self.app.run_server(host=host, port=port, debug=False)
        except Exception as e:
            logger.error(f"Failed to start dashboard: {str(e)}")
            raise

# Global dashboard instance
dashboard = DashboardManager()

# Add this at the end of dashboard.py
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run trading dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    args = parser.parse_args()
    
    dashboard = DashboardManager()
    dashboard.run(host=args.host, port=args.port)