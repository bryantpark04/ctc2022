# Options strategy:

We use the Black-Scholes model to find the difference between the actual and theoretical price of the option. If the actual price is less, we purchase the call option and at the same time sell the underlying stock. This is a married call. If the actual price is greater than the theoretical price, we sell the contract while purchasing the underlying stock. This is a covered call.

Our initial strategy was to use put to call ratios for the first week of the period to make a prediction of the options price by the end of the period. Then, we would sell contracts to create a short straddle around our predicted price. This does not work obviously because we are only using call data.
