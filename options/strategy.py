import csv
from datetime import datetime
import math
import numpy as np
from scipy.stats import norm
"""
## Cornell Trading Competition : Derivatives Case Strategy ##

We have provided you with very basic skeleton code for you to implement your
strategy. The strategy will need to have *at least* two functions. One to read
the options data row-by-row, read_data, and one function that trades,
make_trades.

The Strategy is expected to work by rows: It reads the data for the rows of
market data corresponding for 1 minute, with all the different options being
traded at the time, with read_data() and the executes trades with make_trades().

Please do not modify the existing functions signature and make sure to actually
implement them as we will be using them in the grading of your submission.

This file, along with any other files necessary for the strategy to work, will
need to be submitted so make sure to add comments and make your code easy to
read/understand.

Plagiarism is strictly banned and will result in your team being withdrawn from
the case. By writing your names below you agree to follow the code of conduct.

Please provide the name & emails of your team members:
    * Bryant Park (blp73@cornell.edu)
    * Edward Zhang (ez246@cornell.edu)
    * Jordan Han (jsh323@cornell.edu)
    * Marie Aractingi (ma788@cornell.edu)

Best of Luck!
"""


class Strategy:
    def __init__(self):
        self.bs_vals: list[float] = []
        self.bs_iv_vals: list[float] = []
        self.bin_vals: list[float] = []
        self.put_strikes: list[tuple[float, float]] = [] # (volume, strike price)
        self.call_strikes: list[tuple[float, float]] = []
        self.put_ratios: list[tuple[float, float]] = []
        self.call_ratios: list[tuple[float, float]] = []
        self.first_date: datetime | None = None
        self.price_prediction: float | None = None
        self.trades: list[tuple[str, float, float]] = []

    """
    read_data:
        Function that is responsible for providing the strategy with market data.

    args:
        row_vals - An array of array of strings that represents the different
        values in the rows of the raw csv file of the same format as the raw csv
        file provided to you. The outer array will correspond to an array of
        rows.

    returns:
        Nothing
    """
    def read_data(self, row_vals):
        row = row_vals[-1]
        # unpack data
        current_date = datetime.fromisoformat(row[1].split(' ')[0])
        time_to_maturity = (datetime.fromisoformat(row[3]) - current_date).days / 365
        spy_price = (float(row[15]) + float(row[16])) / 2
        strike_price = float(row[4])
        close_price = float(row[9])
        is_call = row[5] == "C"
        volume = float(row[10])

        # calculate black scholes, append to array
        self.bs_vals.append(self.black_scholes(time_to_maturity, .035, spy_price, strike_price, .282, .0125, is_call))
        # self.bin_vals.append(self.binomial(2000, time_to_maturity, .035, spy_price, strike_price, .282, .0125, is_call))

        if close_price < self.bs_vals[-1]: # buy
            self.trades = [(row[5] + "_" + row[4], 0, int(row[13])), ("underlying", int(row[13]), 0)]
        elif close_price > self.bs_vals[-1]: # sell
            self.trades = [(row[5] + "_" + row[4], int(row[11]), 0), ("underlying", 0, int(row[11]))]
        else:
            self.trades = []

        """ This needs puts to work oops
        # #### DEBUG
        # print(f"{self.bs_vals[-1]=}")
        # print(f"{self.put_strikes=}")
        # print(f"{self.call_strikes=}")
        # print(f"{self.trades=}")

        # put-call ratio stuff
        volume = float(row[10])
        if is_call:
            self.call_strikes.append((volume, strike_price))
            self.call_ratios.append((volume, strike_price / self.bs_vals[-1]))
        else:
            print("Append")
            self.put_strikes.append((volume, strike_price))
            self.put_ratios.append((volume, strike_price / self.bs_vals[-1]))

        if not self.first_date:
            self.first_date = current_date
        # elif (current_date - self.first_date).days < 7:
        #     pass # background strategy
        elif (current_date - self.first_date).days >= 1:
            print("SDFJHKDSHFD")
            weighted_put_strike_avg = np.mean(v * p for v, p in self.put_strikes)
            weighted_call_strike_avg = np.mean(v * p for v, p in self.call_strikes)
            put_call_ratio = (sum(v for v, r in self.call_ratios) * np.mean(r for v, r in self.call_ratios)) / (sum(v for v, r in self.put_ratios) * np.mean(r for v, r in self.put_ratios))
            price_prediction = (put_call_ratio - 1) * (weighted_put_strike_avg - weighted_call_strike_avg) + weighted_call_strike_avg

            # make trades
            if is_call and strike_price >= price_prediction + 1:
                self.trades = [(row[3], volume, 0)]
            elif not is_call and strike_price <= price_prediction - 1:
                self.trades = [(row[3], volume, 0)]
            else:
                # self.trades = [("underlying", 0, 0)]
                self.trades = []
        """


    """
    make_trades:
        Function that tells the exchange whether, and at what quantity, the
        strategy wants to buy or sell, go long or short, the option.

    args:
        None

    returns:
        An array of triples (str(strike_expiry date), sell, buy) where sell, buy
        are float values that give how much quantity of contract options you
        want to sell and buy at bid & ask prices respectively and the strike
        price and expiry date to differentiate the option. Strike price is a
        numeric value and expiry date is a string and of the same format as in
        the raw csv file. Sell & buy may not be higher than ask & bid size at
        the given time. The value should be 0 for buy or sell if you want no
        activity on that side.

        You can buy/sell underlying stock by the same as above but rather than
        the first element be str(strike)+str(expiry date) we have the word
        'underlying'
    """
    def make_trades(self):
        return self.trades

    """
    N: number of iterations in binomial tree
    tau: Time to maturity
    r: risk free rate (3.7%)?
    S: Stock price
    K: Strike price
    v: Volatility (Standard deviation of sstock returns) 14.79%
    q: dividend yield rate
    is it a call? (0 or 1) value 0 if put
    """
    def black_scholes(self, tau, r, S, K, v, q, call):
        normalCDF=norm.cdf
        mult = 2 * (call - 1.5)

        d1 = (np.log(S / K) + (r - q + (.5 * (v ** 2))) * tau) / (v * math.sqrt(tau))
        d2 = d1 - (v * math.sqrt(tau))
        return mult * (S * (math.e ** (-q * tau)) * normalCDF(mult * d1) - (
                    normalCDF(mult * d2) * K * math.e ** (-r * tau)))

    ## N: number of iterations in binomial tree
    ## T: Time to maturity
    ## r: risk free rate (3%)?
    ## S: Stock price
    ## K: Strike price
    ## v: Volatility (Standard deviation of sstock returns) 14.79%
    ## q: dividend yield rate
    ## is it a call? (0 or 1) value 0 if put
    def binomial(self, N, T, r, S, K, v, q, call):
        dt = T / N
        u = math.e ** (v * math.sqrt(dt))
        d = 1 / u
        p = ((math.e ** ((r - q) * dt)) - d) / (u - d)
        dsct = math.e ** (-r * dt)

        mult = 2 * (call.value - 1.5)

        stockPrice = np.zeros(2 * N + 1)
        optionPrice = np.zeros(2 * N + 1)

        stockPrice[N] = S

        for j in range(1, N + 1):
            stockPrice[j + N] = stockPrice[N + j - 1] * u
            stockPrice[N - j] = stockPrice[N - j + 1] * d

        for j in range(-N, N + 1, 2):
            optionPrice[j + N] = max(mult * (stockPrice[j + N] - K), 0)

        for n in range(N - 1, -1, -1):
            for j in range(-n, n + 1, 2):
                optionPrice[j + N] = max(dsct * (p * optionPrice[j + N + 1] + (1 - p) * optionPrice[j + N - 1]),
                                         mult * (stockPrice[j + N] - K))

        return optionPrice[N]
