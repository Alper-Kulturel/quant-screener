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
    return 100 - 100/(1 + up/dn)

def analyze(t):
    try:
        df = yf.Ticker(t).history(period="2y", interval="1d")
        if df.empty or len(df) < 260: return None
        c = df["Close"]; rets = c.pct_change().dropna()
        return {"ticker":t,"price":round(float(c.iloc[-1]),2),
                "momentum_12_1":round(float((c.iloc[-21]/c.iloc[-252]-1)*100),2),
                "vol_30d":round(float(rets.tail(30).std()*np.sqrt(252)*100),2),
                "rsi_14":round(float(rsi(c).iloc[-1]),1),
                "dist_ma50":round(float((c.iloc[-1]/c.rolling(50).mean().iloc[-1]-1)*100),2),
                "sharpe_1y":round(float((rets.tail(252).mean()/rets.tail(252).std())*np.sqrt(252)),2)}
    except Exception: return None

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
