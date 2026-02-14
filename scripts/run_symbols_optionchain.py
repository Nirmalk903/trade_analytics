from get_optionchain_kite import group_option_chain_by_expiry, fetch_ltp_map, save_option_chain_json
from kiteconnect import KiteConnect
import os,json
from pathlib import Path

# Preset symbol list (common liquid names)
symbols = [
    'RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK','LT','SBIN','KOTAKBANK','BHARTIARTL','ITC'
]

session_path = os.path.expanduser('~/.kite_session.json')
if not os.path.exists(session_path):
    print('No session file at', session_path)
    raise SystemExit(1)

s=json.load(open(session_path,'r',encoding='utf8'))
api_key=os.environ.get('KITE_API_KEY') or s.get('api_key')
access_token=s.get('access_token')
if not api_key or not access_token:
    print('Missing api_key or access_token')
    raise SystemExit(1)

kc=KiteConnect(api_key=api_key)
kc.set_access_token(access_token)
print('Fetching instruments...')
instruments = kc.instruments('NFO')
print('Fetched instruments:', len(instruments))

out_dir = Path('./OptionChainJSON_Kite')
out_dir.mkdir(parents=True, exist_ok=True)

for symbol in symbols:
    try:
        print('\nProcessing', symbol)
        group = group_option_chain_by_expiry(instruments, symbol)
        if not group:
            print('  No option instruments for', symbol)
            continue
        # pick nearest expiry in same month if possible
        now = None
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
        except Exception:
            now = None
        # choose last weekday of current month or first available
        target_expiry = next(iter(group.keys()))
        chosen = target_expiry
        chain = group[chosen]
        tokens=[]
        for st, rec in chain.items():
            for ot in ('CE','PE'):
                v = rec.get(ot)
                if v and v.get('instrument_token'):
                    try:
                        tokens.append(int(v['instrument_token']))
                    except Exception:
                        pass
        tokens = sorted(set(tokens))
        print('  Collected', len(tokens), 'tokens')
        ltp_map = fetch_ltp_map(kc, tokens, chunk_size=100, pause=0.25)
        save_option_chain_json(out_dir, symbol, chosen, chain, ltp_map)
        print('  Saved', symbol)
    except Exception as e:
        print('  Error for', symbol, e)
