FINAL PROJECT EMMA MACHADO

ZILLOW PRIZE

https://github.com/emachadoc/zillow-competition

Datos de la competición en:

https://bit.ly/2YgUfFc

Zillow is an american real state company, 'Zestimates' are their estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property with a 5% median margin of error.

In this competition, Zillow is asking to predict, not the sale price, but the log-error between their 'Zestimate' and the actual sale price, given all the features of a home:

logerror = log(Zestimate) − log(SalePrice)

As initial data, we are provided with a list of 2.99 M real estate properties in three counties (Los Angeles, Orange and Ventura, California) data in 2016 and 2017.

We are also asked to predict log-error at 6 time points for all properties: October 2016 (201610), November 2016 (201611), December 2016 (201612), October 2017 (201710), November 2017 (201711), and December 2017 (201712).

The tax assessment fields in 2016 data reflect taxes as they were assessed and levied in 2015 and the tax assessment fields in 2017 data reflect taxes as they were assessed and levied in 2016, but we should not use the 2016 tax values when predicting log errors against the 2016 log errors.

Why did Zillow pick the log error instead of an absolute error metric such as RMSE?

Home sale prices have a right skewed distribution and are also strongly heteroscedastic, so they need to use a relative error metric instead of an absolute metric to ensure valuation models are not biased towards expensive homes. A relative error metric like the percentage error or log ratio error avoids these problems. The log error is free of bias problem and when using the natural logarithm, errors close to 1 approximate percentage errors quite closely.

I chose to use CatBoost library. Catboost is a machine learning algorithm that uses gradient boosting on decision trees.
It handles categorical (CAT) data automatically and provides best-in-class accuracy without extensive data training.
It is competitive at performance with any other machine learning methods, besides, it is robust, reduces the need for extensive hyper-parameter tuning and it is easy to use.

I followed the next steps during the process:

- Exploratory analysis
- Preparing dataset: Training and test files
- Memory management optimization
- Feature engineering
- Training model
- Getting submission file

Obviusly, my submission is not in the top 5, even in the top 1000... A 0.06478 scoring , which is 0.0016 far from the 0.06318 scoring of the first team in the competition, leads me to the 2198 - 2206 range.