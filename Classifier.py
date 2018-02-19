#/usr/bin/python

import numpy as np
import pandas as pd

import DecisionTreeTools


df = DecisionTreeTools.import_dataset()
print(df.head(10))
