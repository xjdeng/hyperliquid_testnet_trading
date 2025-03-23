from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from google.colab import userdata
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import time
from duckduckgo_search import DDGS
import enum
import json
from typing_extensions import TypedDict
import google.generativeai as genai
from decimal import Decimal, ROUND_DOWN
from math import floor

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Logging is working")

class Sentiment(enum.Enum):
  negative = "negative"
  neutral = "neutral"
  positive = "positive"

class Confidence(enum.Enum):
  low = "low"
  medium = "medium"
  high = "high"

class Result(TypedDict):
    sentiment: Sentiment
    confidence: Confidence
    explanation: str

class MyAgent:

    def __init__(self, hypernet_private_key, google_api_key, gemini_model = "gemini-1.5-flash-latest"):
        genai.configure(api_key = google_api_key)
        self.llm = genai.GenerativeModel(gemini_model)
        self.account = Account.from_key(hypernet_private_key)
        self.address = self.account.address
        self.exchange = Exchange(self.account, constants.TESTNET_API_URL)
        self.info = Info(constants.TESTNET_API_URL)

    def get_markets(self, top_n_most_liquid = 50):
        meta, asset_ctxs = self.info.meta_and_asset_ctxs()
        market_stats = [
            {
                "name": m["name"],
                "dayNtlVlm": float(a["dayNtlVlm"]),
                "openInterest": float(a["openInterest"]),
                "impactPxs": a.get("impactPxs", None)
            }
            for m, a in zip(meta["universe"], asset_ctxs)
        ]

        # Sort by 24h volume (descending) and take top 10
        top_markets_all = sorted(market_stats, key=lambda x: x["dayNtlVlm"], reverse=True)
        top_markets = top_markets_all[:top_n_most_liquid]
        return top_markets
    
    def _get_hourly_candles(self, coin, lookback_hours=24):
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=lookback_hours)
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(now.timestamp() * 1000)

        candles = self.info.candles_snapshot(
            name=coin,
            interval="1h",
            startTime=start_ts,
            endTime=end_ts
        )
        return candles
    
    # Robust wrapper to fetch candles with unlimited retries for HTTP 429 errors
    def _safe_get_candles(self, coin, initial_delay=1, max_delay=60, lookback_hours = 24):
        delay = initial_delay
        attempt = 0
        while True:
            try:
                candles = self._get_hourly_candles(coin, lookback_hours)
                return candles
            except Exception as e:
                # Try to extract error code from exception attributes or message
                code = None
                if hasattr(e, "status_code"):
                    code = e.status_code
                elif isinstance(e, tuple) and len(e) > 0:
                    code = e[0]
                elif "429" in str(e):
                    code = 429

                if code == 429:
                    attempt += 1
                    print(f"Rate limit hit for {coin}, retrying in {delay} seconds... (attempt {attempt})")
                    time.sleep(delay)
                    # Exponential backoff with cap
                    delay = min(delay * 2, max_delay)
                else:
                    raise e
                
    # Prepare OHLC points: extract open, high, low, close from each candle (4 points per candle)
    def _prepare_ohlc_points(self, candles):
        all_points = []
        for candle in candles:
            try:
                o = float(candle["o"])
                h = float(candle["h"])
                l = float(candle["l"])
                c = float(candle["c"])
                all_points.extend([o, h, l, c])
            except Exception as e:
                print("Error processing candle:", candle, e)
        return np.array(all_points)

    # Compute regression slope and standard deviation of residuals, then return slope/std
    def _compute_slope_std_ratio(self, y_vals):
        x_vals = np.arange(len(y_vals))
        A = np.vstack([x_vals, np.ones_like(x_vals)]).T
        slope, intercept = np.linalg.lstsq(A, y_vals, rcond=None)[0]
        y_pred = slope * x_vals + intercept
        residuals = y_vals - y_pred
        std_dev = np.std(residuals)
        if std_dev == 0:
            return float("-inf")
        return slope / std_dev

    # Main loop: iterate over the top_markets (top 100 securities) and compute the regression-based score.

    def calculate_signal_scores(self, top_markets, lookback_hours = 24):
        signal_scores = []

        for market in top_markets:
            name = market["name"]
            try:
                candles = self._safe_get_candles(name, lookback_hours = lookback_hours)
                if len(candles) < lookback_hours:
                    print(f"Not enough data for {name}.")
                    continue  # skip if insufficient data
                points = self._prepare_ohlc_points(candles)  # 96 points total (24 bars * 4 OHLC points)
                if len(points) == 0:
                    continue
                score = self._compute_slope_std_ratio(points)
                signal_scores.append({
                    "name": name,
                    "score": score
                })
            except Exception as e:
                print(f"Error processing {name}: {e}")

        # Sort the watchlist by the computed score (slope / volatility ratio) in descending order
        ranked_markets = sorted(signal_scores, key=lambda x: x["score"], reverse=True)
        print("Ranked Markets (top 10):")
        for i, market in enumerate(ranked_markets[:10], 1):
            print(f"{i}. {market['name']} - Score: {market['score']:.4f}")
        return ranked_markets

    def _get_crypto_news_text(self, ticker, max_results=50):
        query = f"{ticker} crypto news"
        results = []

        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                title = r.get("title", "").strip()
                body = r.get("body", "").strip()
                url = r.get("url", "").strip()
                if title:
                    results.append(f"{title}\n{body}\n{url}\n")

        return "\n".join(results)
    
    def _predict_sentiment_llm(self, txt, crypto):
        prompt = f"""
        Gauge the overall sentiment of the Duckduckgo Search Results below about a particular cryptocurrency: {crypto}.
        If there are both positive and negative sentiment articles in the mix, pay attention to which side seems to be more dominant.
        Weight more recent news more than older news.
        If neither positive or negative news is clearly dominant, then the sentiment is neutral
        Please also state your confidence in the prediction and give an explanation for the type you chose and your confidence in it.

        Search Results:
        ---
        {txt}
        ---
        """
        result = self.llm.generate_content(prompt,
                                        generation_config = genai.GenerationConfig(
                                            response_mime_type="application/json", 
                                            response_schema=Result
                                        ))
        return json.loads(result.to_dict()['candidates'][0]['content']['parts'][0]['text'])

    def _predict_sentiment(self, crypto):
        news = self._get_crypto_news_text(crypto)
        return self._predict_sentiment_llm(news, crypto) 

    def get_final_picks(self, original_list, min_ratio = 0, n = 10):
        resulting_list = []
        for coin in original_list:
            if coin['score'] < min_ratio:
                continue
            sentiment = self._predict_sentiment(coin['name'])
            if sentiment['sentiment'] != "negative":
                resulting_list.append(coin['name'])
            if len(resulting_list) >= n:
                break
        missing = n - len(resulting_list)
        for _ in range(missing):
            resulting_list.append("USDC")
        return resulting_list

    def _adjust_order_size(self, coin: str, proposed_size: float) -> float:
        """
        Retrieves the minimum order increment for the given coin from metadata,
        then rounds the proposed_size down to the nearest multiple of that increment.
        
        Returns the adjusted order size.
        """
        try:
            meta = self.info.meta()
        except Exception as e:
            logging.error(f"Error retrieving metadata: {e}")
            return proposed_size
        universe = meta.get("universe", [])
        for asset in universe:
            if asset.get("name", "").upper() == coin.upper():
                sz_decimals = asset.get("szDecimals", 8)
                min_increment = 1 / (10 ** sz_decimals)
                # Floor the proposed size to a multiple of min_increment.
                adjusted_size = floor(proposed_size / min_increment) * min_increment
                logging.debug(f"For {coin}: proposed_size={proposed_size} min_increment={min_increment:.8f} adjusted_size={adjusted_size}")
                return adjusted_size
        logging.error(f"Coin {coin} not found in metadata. Using proposed size {proposed_size}.")
        return proposed_size

    def sell_all_to_usdc(self):
        """
        Closes all open positions for every asset held by the account.
        For each unique coin in info.user_state(), it calls exchange.market_close(coin)
        to liquidate the position.
        """
        address = self.exchange.wallet.address
        state = self.info.user_state(address)
        positions = state.get("assetPositions", [])
        if not positions:
            logging.info("No positions to close.")
            return

        # Build a set of unique coins with positions.
        coins = set()
        for pos in positions:
            coin = pos.get("position", {}).get("coin")
            if coin:
                coins.add(coin)

        for coin in coins:
            logging.info(f"Attempting to market close all positions for {coin}.")
            order_result = self.exchange.market_close(coin)
            if order_result and order_result.get("status") == "ok":
                for status in order_result["response"]["data"]["statuses"]:
                    try:
                        filled = status["filled"]
                        logging.info(f"Market close order #{filled['oid']} for {coin} filled {filled['totalSz']} @{filled['avgPx']}")
                    except KeyError:
                        logging.error(f"Error closing position for {coin}: {status.get('error')}")
            else:
                logging.error(f"Market close failed for {coin}: {order_result}")
            time.sleep(2)  # small pause between orders

    def rebalance_equal_weight(self, final_picks, slippage=0.01):
        """
        Transforms your current portfolio into the target equal‑weighted portfolio (final_picks)
        using the minimum number of net trades, and without using margin.

        Steps:
        1. Retrieve current portfolio state:
            - Net worth is taken from crossMarginSummary["accountValue"].
            - Available cash is taken from state["withdrawable"].
            - Current positions are computed using current mid prices.
        2. Compute target (equal‑weight) allocation:
            For each coin in final_picks (duplicates allowed),
                desired allocation (in USDC) = (frequency of coin in final_picks) × (net_worth / total_slots)
        3. For coins held but not in the target, sell their entire positions.
        4. For coins in the target:
            a. Compute diff = (desired allocation) – (current market value).
            b. If diff is negative (over‑allocated), place one sell order for the excess amount.
                (This will generate additional cash.)
            c. If diff is positive (under‑allocated), place one buy order for the difference, using available cash.
                (If available cash is insufficient, buy as much as possible.)
            In each case, order size = (abs(diff) / mid price), adjusted to the coin’s minimum increment.
        5. Any leftover cash remains in USDC.
        6. The final state should have a USDC balance as close to 0 as possible.
        """
        address = self.exchange.wallet.address
        state = self.info.user_state(address)
        
        # 1. Retrieve net worth and available cash.
        cross_margin = state.get("crossMarginSummary", {})
        try:
            net_worth = float(cross_margin.get("accountValue", 0))
        except Exception as e:
            raise ValueError("Error parsing net worth: " + str(e))
        if net_worth <= 0:
            raise ValueError("Net worth is zero; nothing to rebalance.")
        
        try:
            available_cash = float(state.get("withdrawable", 0))
        except Exception as e:
            available_cash = 0.0
            logging.error("Error parsing available cash: " + str(e))
        
        # 2. Get current positions and compute their market values.
        positions = state.get("assetPositions", [])
        mids = self.info.all_mids()
        current_holdings = {}  # coin -> total USDC value of current holding
        for pos in positions:
            coin = pos.get("position", {}).get("coin")
            if not coin:
                continue
            try:
                size = float(pos.get("position", {}).get("szi", 0))
                mid_price = float(mids.get(coin, 0))
                value = size * mid_price
            except Exception as e:
                logging.error(f"Error processing position for {coin}: {e}")
                continue
            current_holdings[coin] = current_holdings.get(coin, 0) + value

        logging.info(f"Net worth: ${net_worth:.2f}. Available cash: ${available_cash:.2f}.")
        logging.info(f"Current holdings (USDC value): {current_holdings}")
        
        # 3. Compute target allocation for each coin in final_picks.
        total_slots = len(final_picks)
        if total_slots == 0:
            logging.info("No target picks provided.")
            return
        target_each = net_worth / total_slots
        desired_allocation = {}  # coin -> desired USDC value
        for coin in final_picks:
            desired_allocation[coin] = desired_allocation.get(coin, 0) + target_each
        logging.info(f"Desired allocation (USDC): {desired_allocation}")
        
        # 4a. For coins held but not in target, sell entire positions.
        for coin in list(current_holdings.keys()):
            if coin not in desired_allocation:
                logging.info(f"Coin {coin} is held but not desired. Selling entire position.")
                order_result = self.exchange.market_close(coin)
                if order_result and order_result.get("status") == "ok":
                    logging.info(f"Market close for {coin} successful: {order_result}")
                    # Simulate adding cash: assume proceeds equal current holding value.
                    available_cash += current_holdings[coin]
                else:
                    logging.error(f"Market close failed for {coin}: {order_result}")
                current_holdings.pop(coin)
                time.sleep(2)
        
        # 4b. For each target coin, compute net difference and trade.
        for coin, desired_value in desired_allocation.items():
            current_value = current_holdings.get(coin, 0)
            diff = desired_value - current_value  # positive means under‑allocated (need to buy)
            try:
                mid_price = float(mids.get(coin, 0))
                if mid_price <= 0:
                    logging.error(f"No valid mid price for {coin}; skipping trade.")
                    continue
            except Exception as e:
                logging.error(f"Error retrieving mid price for {coin}: {e}")
                continue
            
            if diff < 0:
                # Over‑allocated: sell the excess.
                raw_size = abs(diff) / mid_price
                adjusted_size = self._adjust_order_size(coin, raw_size)
                if adjusted_size <= 0:
                    logging.error(f"Adjusted order size for {coin} is zero; skipping sell trade.")
                    continue
                logging.info(f"SELL {coin}: excess value ${abs(diff):.2f} -> order size {adjusted_size:.8f} (mid price ${mid_price:.2f})")
                order_result = self.exchange.order(
                    name=coin,
                    is_buy=False,
                    sz=adjusted_size,
                    limit_px=mid_price,
                    order_type={"limit": {"tif": "Ioc"}},
                    reduce_only=True
                )
                if order_result and order_result.get("status") == "ok":
                    logging.info(f"Sell order for {coin} executed: {order_result}")
                    # Simulate adding cash from sale.
                    available_cash += adjusted_size * mid_price
                else:
                    logging.error(f"Sell order for {coin} failed: {order_result}")
            else:
                # Under‑allocated: need to buy additional value.
                # Use available_cash; if not enough, buy as much as possible.
                cash_to_use = min(diff, available_cash)
                if cash_to_use < diff:
                    logging.info(f"Insufficient cash for {coin}: need ${diff:.2f} but have only ${available_cash:.2f}. Will buy with available cash.")
                raw_size = cash_to_use / mid_price
                adjusted_size = self._adjust_order_size(coin, raw_size)
                if adjusted_size <= 0:
                    logging.error(f"Adjusted order size for {coin} is zero; skipping buy trade.")
                    continue
                logging.info(f"BUY {coin}: under-allocated by ${diff:.2f}, using ${cash_to_use:.2f} -> order size {adjusted_size:.8f} (mid price ${mid_price:.2f})")
                order_result = self.exchange.market_open(coin, True, adjusted_size, None, slippage)
                if order_result and order_result.get("status") == "ok":
                    logging.info(f"Market open order for {coin} executed: {order_result}")
                    available_cash -= adjusted_size * mid_price
                else:
                    logging.error(f"Market open order for {coin} failed: {order_result}")
            time.sleep(2)
        
        # 6. Log final state.
        final_state = self.info.user_state(address)
        logging.info(f"Rebalance complete. Final state for {address}: {final_state}")


    def display_positions(self):
        """
        Retrieves and displays the current portfolio positions line by line.
        For each position, it shows:
        - Ticker
        - Number of Shares held
        - Price per share (current mid price)
        - Market value (shares * price per share)
        
        A line for USDC is included (using the 'withdrawable' cash, with price assumed as $1.00).
        Finally, the total portfolio net worth (as defined by marginSummary["accountValue"]) is printed.
        """
        address = self.exchange.wallet.address
        state = self.info.user_state(address)
        
        # Retrieve net worth from marginSummary ("accountValue").
        margin_summary = state.get("marginSummary", {})
        try:
            net_worth = float(margin_summary.get("accountValue", 0))
        except Exception as e:
            net_worth = 0.0
            print("Error parsing net worth from marginSummary:", e)
        
        # Retrieve available USDC from 'withdrawable'.
        try:
            usdc_balance = float(state.get("withdrawable", 0))
        except Exception as e:
            usdc_balance = 0.0
            print("Error parsing USDC balance from withdrawable:", e)
        
        # Retrieve current mid prices.
        mids = self.info.all_mids()
        
        # Prepare positions details.
        positions = state.get("assetPositions", [])
        positions_details = []
        for pos in positions:
            p = pos.get("position", {})
            ticker = p.get("coin", "Unknown")
            try:
                shares = float(p.get("szi", 0))
            except Exception as e:
                shares = 0.0
                print(f"Error parsing shares for {ticker}: {e}")
            try:
                price = float(mids.get(ticker, 0))
            except Exception as e:
                price = 0.0
                print(f"Error retrieving price for {ticker}: {e}")
            market_value = shares * price
            positions_details.append((ticker, shares, price, market_value))
        
        # Print header.
        header = f"{'Ticker':<10} {'Shares':>15} {'Price':>15} {'Market Value':>20}"
        print(header)
        print("-" * len(header))
        
        # Print USDC line (assuming $1 per USDC).
        usdc_line = f"{'USDC':<10} {usdc_balance:>15.8f} {1.00:>15.2f} {usdc_balance:>20.2f}"
        print(usdc_line)
        
        # Print each position.
        for ticker, shares, price, market_value in positions_details:
            print(f"{ticker:<10} {shares:>15.8f} {price:>15.2f} {market_value:>20.2f}")
        
        print("-" * len(header))
        # Print total portfolio net worth from marginSummary.
        print(f"{'Total Portfolio Net Worth:':<10} {net_worth:>15.2f}")



    def run(self, watchlist = None,
            top_n_most_liquid = 50,
            n_hours_lookback = 24,
            n_picks = 10,
            hours_holding_period = 1,
            number_holding_periods = 9999999999999):
        if not watchlist:
            watchlist = self.get_markets(top_n_most_liquid)
        for _ in range(number_holding_periods):
            
            ranked_markets = self.calculate_signal_scores(watchlist, n_hours_lookback)
            final_picks = self.get_final_picks(ranked_markets, n = n_picks)
            print("Current Portfolio:\n\n")
            self.display_positions()
            self.rebalance_equal_weight(final_picks)
            print("Portfolio after trading:\n\n")
            self.display_positions()
            waittime = 3600*hours_holding_period
            print(f"\n\n\nWaiting {waittime} seconds before trading agian.")
            time.sleep(waittime)