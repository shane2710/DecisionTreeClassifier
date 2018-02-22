# Decision Tree Machine Learning Classifier

## Implementation of a greedy Decision Tree Classifier from scratch using pandas for efficient data handling, multi-way splits on discrete feature sets, and maximization of an information gain cost function for optimization.

This project was built using 'heart.data' included in the above repository,
which contains example medical information on patients who do or do not have
heart disease.  The trained classifier is able to identify new data inputs as
either positive or negative for heart disease presence.

## Example output of a trained tree with a minimum of 25 data points at each split:

```
[Thal {n} = 3.000]
 [ChestPain {n} = 1.000]
  [1.0]
 [ChestPain {n} = 2.000]
  [STDepressionvsRest {c} < 1.000]
   [SerumCholestoral {c} < 319.000]
    [1.0]               # <---    when both children are one value, classification
    [1.0]               #         probability is 50/50 for positive / negative
   [2.0]
 [ChestPain {n} = 3.000]
  [Sex {n} = 0.000]
   [1.0]
  [Sex {n} = 1.000]
   [MaxHeartRate {c} < 162.000]
    [2.0]				# <---	first child contains prediction for < 162
    [1.0]				# <---	second child for >= 162
 [ChestPain {n} = 4.000]
  [VesselsColoredbyFluoroscopy {c} < 1.000]
   [Age {c} < 55.000]
    [1.0]
    [2.0]
   [1.0]
[Thal {n} = 6.000]
 [1.0]
[Thal {n} = 7.000]
 [ChestPain {n} = 1.000]
  [2.0]
 [ChestPain {n} = 2.000]
  [1.0]
 [ChestPain {n} = 3.000]
  [2.0]
 [ChestPain {n} = 4.000]
  [STDepressionvsRest {c} < 0.800]
   [2.0]
   [1.0]
```
