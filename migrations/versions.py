"""
Initial database schema for the trading system.
Revision ID: 001_initial_schema
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create OHLCV table
    op.create_table(
        'ohlcv',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('open', sa.Float, nullable=False),
        sa.Column('high', sa.Float, nullable=False),
        sa.Column('low', sa.Float, nullable=False),
        sa.Column('close', sa.Float, nullable=False),
        sa.Column('volume', sa.Float, nullable=False),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
    )
    
    op.create_index('idx_ohlcv_symbol_timestamp', 'ohlcv', ['symbol', 'timestamp'])
    op.create_index('idx_ohlcv_timeframe', 'ohlcv', ['timeframe'])
    
    # Create trades table
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('order_id', sa.String(50), unique=True),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('amount', sa.Float, nullable=False),
        sa.Column('price', sa.Float, nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('params', sa.JSON),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
    )
    
    op.create_index('idx_trades_symbol', 'trades', ['symbol'])
    op.create_index('idx_trades_timestamp', 'trades', ['timestamp'])
    
    # Create positions table
    op.create_table(
        'positions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('amount', sa.Float, nullable=False),
        sa.Column('entry_price', sa.Float, nullable=False),
        sa.Column('current_price', sa.Float, nullable=False),
        sa.Column('unrealized_pnl', sa.Float, nullable=False),
        sa.Column('realized_pnl', sa.Float, nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow),
    )
    
    op.create_index('idx_positions_symbol', 'positions', ['symbol'])
    op.create_index('idx_positions_active', 'positions', ['active'])
    
    # Create portfolio_history table
    op.create_table(
        'portfolio_history',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('equity', sa.Float, nullable=False),
        sa.Column('margin_used', sa.Float),
        sa.Column('margin_free', sa.Float),
        sa.Column('positions_value', sa.Float),
        sa.Column('daily_pnl', sa.Float),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
    )
    
    op.create_index('idx_portfolio_timestamp', 'portfolio_history', ['timestamp'])
    
    # Create risk_metrics table
    op.create_table(
        'risk_metrics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('var_95', sa.Float),
        sa.Column('cvar_95', sa.Float),
        sa.Column('sharpe_ratio', sa.Float),
        sa.Column('sortino_ratio', sa.Float),
        sa.Column('max_drawdown', sa.Float),
        sa.Column('current_drawdown', sa.Float),
        sa.Column('portfolio_beta', sa.Float),
        sa.Column('correlation_matrix', sa.JSON),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
    )
    
    op.create_index('idx_risk_timestamp', 'risk_metrics', ['timestamp'])

def downgrade():
    op.drop_table('risk_metrics')
    op.drop_table('portfolio_history')
    op.drop_table('positions')
    op.drop_table('trades')
    op.drop_table('ohlcv')