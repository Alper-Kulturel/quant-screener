#!/usr/bin/env python3
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd, numpy as np
from datetime import datetime, timezone
import yfinance as yf
from universe import UNIVERSE

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean()
    dn = -d.clip(upper=0).rolling(n).mean()
    # A window with no losing days has dn == 0. With gains that is a genuine
    # RSI of 100, but a completely flat window leaves 0/0 -> NaN, and a NaN
    # rsi_14 silently drops out of the RSI rank (rank() skips NaN) instead of
    # scoring as neutral. Pin the flat case to 50 and the up-only case to 100.
    rs = up / dn
    rs = rs.mask((dn == 0) & (up == 0), 1.0)    # flat window -> RSI 50
    rs = rs.mask((dn == 0) & (up > 0), np.inf)  # gains only  -> RSI 100
    return 100 - 100/(1 + rs)

def analyze(t):
    # Fetch and compute are reported separately. A delisted ticker or a dropped
    # connection is routine and just drops out of the screen, but a failure
    # inside the indicator maths is a bug, and a bare `except Exception:
    # return None` used to make the two indistinguishable.
    try:
        df = yf.Ticker(t).history(period="2y", interval="1d")
    except Exception as exc:
        print(f"warn: {t}: fetch failed ({exc})", file=sys.stderr)
        return None
    if df.empty or len(df) < 260: return None
    try:
        c = df["Close"]; rets = c.pct_change().dropna()
        return {"ticker":t,"price":round(float(c.iloc[-1]),2),
                "momentum_12_1":round(float((c.iloc[-21]/c.iloc[-252]-1)*100),2),
                "vol_30d":round(float(rets.tail(30).std()*np.sqrt(252)*100),2),
                "rsi_14":round(float(rsi(c).iloc[-1]),1),
                "dist_ma50":round(float((c.iloc[-1]/c.rolling(50).mean().iloc[-1]-1)*100),2),
                "sharpe_1y":round(float((rets.tail(252).mean()/rets.tail(252).std())*np.sqrt(252)),2)}
    except Exception as exc:
        print(f"warn: {t}: indicator computation failed ({exc})", file=sys.stderr)
        return None

def rank(df):
    d = df.copy()
    d["r_mom"]=d["momentum_12_1"].rank(ascending=False)
    d["r_vol"]=d["vol_30d"].rank(ascending=True)
    d["r_rsi"]=(d["rsi_14"]-50).abs().rank(ascending=True)
    d["r_dist"]=d["dist_ma50"].rank(ascending=False)
    d["score"]=d[["r_mom","r_vol","r_rsi","r_dist"]].mean(axis=1)
    return d.sort_values("score")

def run():
    rows = [r for r in (analyze(t) for t in UNIVERSE) if r]
    if not rows:
        # An all-tickers-failed run used to reach rank() and die there with a
        # bare KeyError on a column-less frame; bail out somewhere legible.
        sys.exit("no tickers produced data — refusing to write an empty screen")
    df = pd.DataFrame(rows); ranked = rank(df)
    ranked["rank"] = range(1, len(ranked)+1)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ranked.to_csv(f"data/{date}_screener.csv", index=False)
    Path(f"data/{date}_screener.json").write_text(json.dumps({"date":date,"timestamp":ts,"rows":ranked.to_dict("records")}, indent=2))
    longs = ranked.head(5); shorts = ranked.tail(5)
    L = [f"# Quant Screener — {date}","",f"_Generated {ts} · {len(ranked)} tickers_","",
         "## Top 5 long candidates","",
         "| Rank | Ticker | Price | Momentum | Vol | RSI | vs MA50 | Sharpe |","|---|---|---|---|---|---|---|---|"]
    for _, r in longs.iterrows():
        L.append(f"| {r["rank"]} | {r["ticker"]} | {r["price"]} | {r["momentum_12_1"]:+.1f}% | {r["vol_30d"]:.1f}% | {r["rsi_14"]} | {r["dist_ma50"]:+.1f}% | {r["sharpe_1y"]} |")
    L += ["","## Bottom 5 short candidates","",
          "| Rank | Ticker | Price | Momentum | Vol | RSI | vs MA50 | Sharpe |","|---|---|---|---|---|---|---|---|"]
    for _, r in shorts.iterrows():
        L.append(f"| {r["rank"]} | {r["ticker"]} | {r["price"]} | {r["momentum_12_1"]:+.1f}% | {r["vol_30d"]:.1f}% | {r["rsi_14"]} | {r["dist_ma50"]:+.1f}% | {r["sharpe_1y"]} |")
    Path(f"reports/{date}.md").write_text("\n".join(L)+"\n")
    Path("LATEST.md").write_text("\n".join(L)+"\n")
    Path("README.md").write_text(f"# Quant Screener\n\nDaily factor screener over {len(ranked)} US large caps.\n\n**Latest:** [{date}](reports/{date}.md)\n")
    print(f"screener complete: {len(ranked)} tickers, {date}")

if __name__ == "__main__": run()
