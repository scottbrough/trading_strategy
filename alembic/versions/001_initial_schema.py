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
    # Create trades table with IF NOT EXISTS
    op.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        side VARCHAR(4) NOT NULL,
        entry_price FLOAT NOT NULL,
        exit_price FLOAT,
        amount FLOAT NOT NULL,
        entry_time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        exit_time TIMESTAMP WITHOUT TIME ZONE,
        pnl FLOAT,
        status VARCHAR(10) NOT NULL,
        strategy VARCHAR(50),
        parameters JSON,
        PRIMARY KEY (id)
    )
    """)
    
    # Do the same for other tables
    op.execute("""
    CREATE TABLE IF NOT EXISTS ohlcv (
        id SERIAL NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        timeframe VARCHAR(3) NOT NULL,
        open FLOAT NOT NULL,
        high FLOAT NOT NULL,
        low FLOAT NOT NULL,
        close FLOAT NOT NULL,
        volume FLOAT NOT NULL,
        PRIMARY KEY (id)
    )
    """)
    
    op.execute("""
    CREATE TABLE IF NOT EXISTS performance (
        id SERIAL NOT NULL,
        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        equity FLOAT NOT NULL,
        realized_pnl FLOAT NOT NULL,
        unrealized_pnl FLOAT,
        total_trades INTEGER,
        win_rate FLOAT,
        sharpe_ratio FLOAT,
        sortino_ratio FLOAT,
        max_drawdown FLOAT,
        volatility FLOAT,
        PRIMARY KEY (id)
    )
    """)
    
    # Create indices
    op.execute("CREATE INDEX IF NOT EXISTS ix_trades_symbol ON trades (symbol)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_trades_entry_time ON trades (entry_time)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_trades_status ON trades (status)")
    
    op.execute("CREATE INDEX IF NOT EXISTS ix_ohlcv_symbol_tf_ts ON ohlcv (symbol, timeframe, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_performance_timestamp ON performance (timestamp)")