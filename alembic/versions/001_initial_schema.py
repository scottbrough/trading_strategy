"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2023-07-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create trades table
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(4), nullable=False),
        sa.Column('entry_price', sa.Float, nullable=False),
        sa.Column('exit_price', sa.Float),
        sa.Column('amount', sa.Float, nullable=False),
        sa.Column('entry_time', sa.DateTime, nullable=False),
        sa.Column('exit_time', sa.DateTime),
        sa.Column('pnl', sa.Float),
        sa.Column('status', sa.String(10), nullable=False),
        sa.Column('strategy', sa.String(50)),
        sa.Column('parameters', sa.JSON)
    )
    
    # Create OHLCV table
    op.create_table(
        'ohlcv',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('timeframe', sa.String(3), nullable=False),
        sa.Column('open', sa.Float, nullable=False),
        sa.Column('high', sa.Float, nullable=False),
        sa.Column('low', sa.Float, nullable=False),
        sa.Column('close', sa.Float, nullable=False),
        sa.Column('volume', sa.Float, nullable=False)
    )
    
    # Create Performance metrics table
    op.create_table(
        'performance',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('equity', sa.Float, nullable=False),
        sa.Column('realized_pnl', sa.Float, nullable=False),
        sa.Column('unrealized_pnl', sa.Float),
        sa.Column('total_trades', sa.Integer),
        sa.Column('win_rate', sa.Float),
        sa.Column('sharpe_ratio', sa.Float),
        sa.Column('sortino_ratio', sa.Float),
        sa.Column('max_drawdown', sa.Float),
        sa.Column('volatility', sa.Float)
    )
    
    # Add indices for performance
    op.create_index('ix_trades_symbol', 'trades', ['symbol'])
    op.create_index('ix_trades_entry_time', 'trades', ['entry_time'])
    op.create_index('ix_trades_status', 'trades', ['status'])
    
    op.create_index('ix_ohlcv_symbol_tf_ts', 'ohlcv', ['symbol', 'timeframe', 'timestamp'])
    op.create_index('ix_performance_timestamp', 'performance', ['timestamp'])


def downgrade() -> None:
    op.drop_table('performance')
    op.drop_table('ohlcv')
    op.drop_table('trades')