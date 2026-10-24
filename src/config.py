"""Central configuration. All paths and constants in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path.home() / "Library" / "Application Support" / "NeuroTrom-Crypto"
for sub in ("raw/market", "market", "features", "models", "runs", "journal", "state", "logs", "reports"):
    (DATA / sub).mkdir(parents=True, exist_ok=True)

PORT = 8775
HOST = "127.0.0.1"

# Data source settings
PRIMARY_EXCHANGE = "binance"
PRIMARY_MARKET = "spot"
UNIVERSE_SIZE = 60
MAX_BACKFILL_CONCURRENCY = 4
MAX_LIVE_CONCURRENCY = 8

# Model settings
HORIZONS = ("1h", "4h", "24h", "7d")
BASE_PARAMS = dict(
    learning_rate=0.025,
    num_leaves=31,
    min_data_in_leaf=300,
    feature_fraction=0.3,
    bagging_fraction=0.7,
    lambda_l2=100.0,
    max_bin=63,
    extra_trees=True,
    verbose=-1,
    num_threads=0,
)
ROUNDS = 500
N_FOLDS = 5
GAP = 5  # walk-forward gap in bars
COST_BPS = 10  # 10 bps per trade

# Intervals for history
HISTORY_INTERVALS = ["1h", "1d"]
FINE_GRAIN_INTERVALS = ["1m"]
FINE_GRAIN_UNIVERSE_SIZE = 15

# Data retention
CANDLE_RETENTION_DAYS = 365 * 3  # 3 years
