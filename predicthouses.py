import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

# read in file
df = pd.read_csv("USA_Housing.csv")

# delete unneeded column/feature
del df["Address"]

# check to see if there are any null values in the data frame
nulls = pd.isnull(df)
nulls = nulls.sum()

# create list of predictors
predictors = ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms", "Area Population"]
target = "Price"

# splits for cross validation
train, test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 10)

# initialize random forest
randomForest = RandomForestRegressor(min_samples_split=10, random_state=1)
randomForest.fit(train[predictors], train[target])
predictions = randomForest.predict(test[predictors])

# measure accuracy of model
accuracy = r2_score(test[target], predictions)
print(f"Accuracy: {accuracy:>7f}")

# combine predictions and actual values into one data frame 
predictions = pd.Series(predictions, index=test.index) 
combined = pd.concat([test[target], predictions], axis=1)        
# rename columns 
combined.columns = ["actual", "prediction"]

print(combined)