# Hyperliquid Testnet Trading

## Summary

This trading strategy looks at the all of the securities on Hyperliquid, filters them by the most liquid, then the ones with the best momentum relative to volatility and finally the ones with positive or neutral sentiment before equal weighting them.  I'll go into detail about each stage of the strategy below. Any bolded number mentioned in the details below is a parameter that we can later change or optimize in future iterations of this strategy.

## Stage 1: Filter by liquidity

We will get the liquidity of each eligible security and keep the top **100** most liquid securities for consideration.

## Stage 2: Calculate momentum relative to the security's volatility (sharpe ratio)

We calculate the momentum of the security by finding the slope of the regression line through a hypothetical OHLC hourly candlestick chart with respect to the open, high, low, and closing points.  We look at the last 24 hours which means **24 bars** or 24*4 = 96 total points where the regression line will be based off of.  Then we calculate the standard deviation of its hourly returns and we calculate a variation of the "sharpe" ratio by dividing the momentum with the volatility.  Then we sort the securities from highest to lowest sharpe ratios.

## Stage 3: Find the highest sharpe ratio securities that don't have negative sentiment.

We want to build a portfolio of **10** equal-weighted securities but we have one more condition to filter on: Removing any security with negative sentiment.  We go down the list from the last stage sorted from highest to lowest sharpe ratio and get the sentiment of each security.  If the sentiment is positive or neutral, we add it to our final list until we have our target number.  We also don't add any security with a negative sharpe ratio.  If we have less than 10 securities after completing this step, we'll fill the remaining slots with USDC.

As for sentiment, the process is a bit primitive right now: you do a Duckduckgo search for crypto news on the security then use a LLM (Gemini right now) to gauge the overall sentiment of the security from the top 50 search results and the metadata displayed in the Duckduckgo results (with "positive", "neutral", and "negative" as the possible options.)

## Stage 4: Place the trades

To simplfy the trading process, I actually sell all of the current securities to USDC and repurchase the ones in the list at equal weights. (We could optimize this further to reduce the slippage since the list of picks might not change as often which negates the need to liquidate and repurchase the same securities all the time, but I ran out of time here.)  When purchasing the securities, it's important to check the minimum order size for that particular security and round it down to the closest acceptable size before placing the order.

## Stage 5: Repeat the process

Since this strategy looks at an hourly candlestick chart when calculating the regression line and sharpe ratios, I'd recommend waiting at least 1 hour before rerunning and getting a new set of picks.  Since liquidity doesn't change much hour to hour, it may suffice to go back to step 2.

## Further improvements, esp collaboration with other agents (and other things I haven't figured out in the limited amount of time)

- Instead of manually fetching news from Duckduckgo, you could have another agent with the sole job of scraping news and populating a database with the latest news on various securities. You'll need to scrape social media sites like X, Reddit, and various crypto forums fairly frequently (say once every several minutes) while refreshing the web search results less frequently (say every several hours) depending on your trading timeframe.  This agent could serve multiple trading agents.  With this, we'll want to pull from this database in Stage 3.
- You can have another agent manage multiple trading agents like this one. While each agent might have "tunnel vision" on a particular strategy, the manager agent could assess the performance of all of the agents it's overseeing and allocate capital from one agent to another since not every strategy works well in all different market environments and it can employ a data-driven approach to periodically reallocating capital to each.  The manager would either need access to each of its subordinate agents' wallets or each agent should have a "backdoor" function to execute trades that were specifically ordered by the master agent.
- Sometimes, trades don't go thru for various odd reasons and we need a more robust system to make sure they go through (maybe using an LLM to guide the process?)
- For some reason, the program seems to always buy securities on margin even if there is plenty of USDC available. Ideally, we don't want to use margin if it's not necessary.
