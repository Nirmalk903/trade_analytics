
# Copilot Instructions for algotrading

## Project Overview
This codebase is an algorithmic trading platform for Indian options and futures, automating data collection, feature engineering, strategy execution, and reporting. It is designed for robust, repeatable workflows and live trading integration.

## Key Components
- **strategy_position_manager.py**: Core logic for managing option strategies (bull call/bear put spreads, etc.), including detection, risk management, and order placement via KiteConnect.
- **algorithms/**: Advanced models (CUSUM, HMM, clustering, etc.) for signal generation and analytics.
- **feature_engineering.py**: Builds features from raw data, including technical indicators and macro/microeconomic factors.
- **data_download_kite.py / data_download_yf.py / data_download_vbt.py**: Download market data from Kite, Yahoo Finance, and VectorBT.
- **Options_Utility.py**: Option analytics and calculation helpers.
- **results/**: Stores strategy execution reports and logs. Reports are timestamped and organized by strategy type (see `results/kite/strategy_manager/`).

## Developer Workflows
- **Run strategies**: `python strategy_position_manager.py --profit-pct 30 --stop-loss-pct 30 --max-fund-risk 2 --monitor` (add `--dry-run` for simulation).
- **Monitor mode**: Use `--monitor` and optionally `--interval 60` for continuous, interval-based monitoring and auto-exit.
- **Authentication**: Requires Kite API key and access token. Generate tokens via `kite.py`; tokens are stored in `~/.kite_session.json` and loaded via environment or file.
- **Feature engineering**: Run `feature_engineering.py` after data download to update features for analytics and modeling.

## Project-Specific Patterns
- **Strategy Detection**: Strategies are detected by grouping positions by underlying and analyzing option legs (see `detect_strategies()` in `strategy_position_manager.py`).
- **Risk Management**: Targets and stop-losses are set as percentages of max profit/loss and capped by total funds at risk.
- **Data Flow**: Data is downloaded, features are engineered, strategies are executed, and results are logged in `results/`.
- **Config/Secrets**: Use `.env` for API keys and secrets. Session tokens are managed in `~/.kite_session.json`.
- **Live Data**: WebSocket (KiteTicker) and threading are used for live spot price updates during monitoring.
- **Extensibility**: To add new strategies or analytics, follow the patterns in `strategy_position_manager.py` and `algorithms/`.

## Integration Points
- **KiteConnect**: For live trading, order management, and data. See `kite.py` and `strategy_position_manager.py`.
- **VectorBT, YFinance**: For feature engineering and historical data.
- **dotenv**: For environment variable management.

## Example: Entering a Bull Call Spread
- Use `strategy_position_manager.py` logic to detect/manage a bull call spread.
- Place orders via KiteConnect, set targets/stop-loss, and log results in `results/kite/strategy_manager/`.

## Conventions
- All scripts use logging for status and error reporting.
- Results and analytics are stored in dedicated folders by type, with timestamped filenames.
- Option symbols follow the format: `UNDERLYINGYYMONSTRIKECE/PE` (e.g., `NIFTY24JAN24000CE`).
- Use provided command-line examples for consistent execution.

## References
- See `README.md` for high-level workflow and feature engineering details.
- See `strategy_position_manager.py` for strategy logic, risk management, and extensibility patterns.
- See `Options_Utility.py` for option analytics helpers.

---
For new agents: Start by reading `strategy_position_manager.py` and `README.md` to understand the main workflow, conventions, and extension points. Use the provided command-line examples to run and test strategies. Review the results directory for log/report structure.
