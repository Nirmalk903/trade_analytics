# Market Calendar Guide

## Overview
The `calendar_fetcher.py` module fetches upcoming economic events from multiple global regions including Americas, Europe, India, China, and Japan.

## Features
- ✅ Multi-region economic calendar (Americas, Europe, India, China, Japan)
- ✅ Customizable importance levels (high, medium, low)
- ✅ Flexible date ranges (look back/forward)
- ✅ CSV export with timestamps
- ✅ Command-line interface
- ✅ Regional data organization

## Quick Start

### Basic Usage
```bash
# Fetch high-importance events for all regions (next 7 days)
python3 calendar_fetcher.py --save
```

### Advanced Usage

#### Specific Regions
```bash
python3 calendar_fetcher.py --regions Americas Europe India --save
```

#### Multiple Importance Levels
```bash
python3 calendar_fetcher.py --importance high medium --save
```

#### Extended Time Period
```bash
python3 calendar_fetcher.py --days 14 --save
```

#### Save to CSV
```bash
# Save results to CSV files
python3 calendar_fetcher.py --save

# Save to custom directory
python3 calendar_fetcher.py --save --output-dir my_calendar_data
```

#### Combined Options
```bash
# Fetch high & medium importance events for 14 days, save to CSV
python3 calendar_fetcher.py --importance high medium --days 14 --save
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--regions` | Regions to fetch (Americas, Europe, India, China, Japan) | All regions |
| `--importance` | Event importance levels (high, medium, low) | high |
| `--days` | Number of days to look forward | 7 |
| `--save` | Save results to CSV files | False |
| `--output-dir` | Output directory for CSV files | results/economic_calendar |

## Programmatic Usage

### Import Functions
```python
from calendar_fetcher import fetch_and_save, filter_and_save

# Fetch programmatically (returns a dict region -> DataFrame)
calendars = fetch_and_save(from_date=None, to_date=None, regions=['Americas','Europe','India','China','Japan'])

# Access individual region data (if present)
americas_df = calendars.get('Americas')
europe_df = calendars.get('Europe')
india_df = calendars.get('India')
china_df = calendars.get('China')
japan_df = calendars.get('Japan')

# Optionally apply filtering rules and save filtered CSV
filter_and_save()
```

## Regional Coverage

### Americas
- USD (United States)
- CAD (Canada)
- BRL (Brazil)
- MXN (Mexico)

### Europe
- EUR (Eurozone)
- GBP (United Kingdom)
- CHF (Switzerland)
- SEK (Sweden)
- NOK (Norway)
- DKK (Denmark)

### Asia
- **India**: INR
- **China**: CNY, CNH
- **Japan**: JPY

## Output Format

### Console Display
The script displays events organized by region with the following information:
- Date
- Zone/Country
- Currency
- Event name
- Importance level
- Actual value (if released)
- Forecast
- Previous value

### CSV Export
CSV files are saved with timestamps in the format:
```
results/economic_calendar/
├── Americas_calendar_2026-02-08_001510.csv
├── Europe_calendar_2026-02-08_001510.csv
├── India_calendar_2026-02-08_001510.csv
├── China_calendar_2026-02-08_001510.csv
├── Japan_calendar_2026-02-08_001510.csv
└── All_Regions_calendar_2026-02-08_001510.csv
```

## Examples

### Monitor US Fed Events
```bash
python3 market_calendar.py --regions Americas --importance high --days 30
```

### Track Asian Market Events
```bash
python3 market_calendar.py --regions India China Japan --importance high medium --days 14 --save
```

### Weekly Economic Review
```bash
python3 market_calendar.py --importance high --days 7 --save
```

## Integration with Trading Strategy

The economic calendar can be integrated into your trading workflow:

1. **Pre-market Analysis**: Check high-impact events before market open
2. **Risk Management**: Avoid trading during major economic releases
3. **Event-driven Trading**: Position for volatility around scheduled events
4. **Multi-market Coordination**: Track global events affecting correlated markets

## Legacy Function

The original `economic_calendar()` function is maintained for backward compatibility:

```python
from calendar_fetcher import fetch_and_save

df_map = fetch_and_save()
df = df_map.get('India')  # Example: get India region DataFrame
```

## Error Handling

The script includes robust error handling:
- Network connectivity issues
- API rate limits
- Invalid region names
- Missing data

Errors are logged to console with appropriate messages.

## Dependencies

- `investpy`: For fetching economic calendar data
- `pandas`: Data manipulation
- `pendulum`: Date/time handling
- `yfinance`: Stock earnings data
- `requests`: News API integration

## Notes

1. The `investpy` library may show deprecation warnings - these are from the library itself and can be ignored
2. Economic data is fetched in real-time from investing.com
3. Events are displayed in the timezone configured in your system
4. Some regions may have no events during certain periods
5. The script also includes stock earnings calendar and news fetching functions

## Troubleshooting

### No Events Found
- Check internet connectivity
- Verify the date range includes upcoming trading days
- Try expanding importance levels to include 'medium' events

### CSV Files Not Saved
- Ensure you have write permissions in the output directory
- Check if `results/` is in `.gitignore` (files won't be committed to git)

### Import Errors
```bash
# Install required packages
pip install investpy pandas pendulum yfinance requests
```

## Future Enhancements

Potential improvements:
- [ ] Add more regions (Middle East, Africa, Australia)
- [ ] Email/notification alerts for critical events
- [ ] Integration with calendar apps (Google Calendar, Outlook)
- [ ] Historical event impact analysis
- [ ] Machine learning for event importance prediction
- [ ] Real-time updates via WebSocket

## Support

For issues or feature requests, please check the main project README or contact the maintainer.
