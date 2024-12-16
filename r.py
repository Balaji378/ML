# Load necessary libraries
library(dplyr)    # For data manipulation
library(tidyr)    # For data tidying
library(caret)    # For transformations like one-hot encoding

# Load the dataset
data <- read.csv("path_to_your_file/Housing-R.csv")

# 1. Inspect the Dataset
print("Summary of Dataset:")
summary(data)       # Gives an overview of missing values, data types, and statistics
print("Column Names:")
colnames(data)      # Shows the column names
str(data)           # Shows structure, data types, and preview of the data

# 2. Handle Missing Values
# Impute numeric columns with the mean
numeric_cols <- sapply(data, is.numeric)
data[numeric_cols] <- lapply(data[numeric_cols], function(x) {
  ifelse(is.na(x), mean(x, na.rm = TRUE), x)
})

# Impute categorical columns with the mode
categorical_cols <- sapply(data, is.factor) | sapply(data, is.character)
data[categorical_cols] <- lapply(data[categorical_cols], function(x) {
  ifelse(is.na(x), names(sort(table(x), decreasing = TRUE))[1], x)
})

# 3. Convert Categorical Variables
# Convert character columns to factors
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)

# Perform one-hot encoding on categorical columns if needed
dummies <- dummyVars(" ~ .", data = data)
data_transformed <- data.frame(predict(dummies, newdata = data))

# 4. Normalize or Standardize Numeric Columns
data_transformed[numeric_cols] <- lapply(data_transformed[numeric_cols], scale)

# 5. Example of Merging Multiple Data Frames
# Assume data_frame1 and data_frame2 are additional data frames to merge
data_frame1 <- data.frame(ID = 1:5, Value1 = rnorm(5))
data_frame2 <- data.frame(ID = 3:7, Value2 = rnorm(5))

# Merge using a common identifier ('ID' column)
merged_data <- left_join(data_frame1, data_frame2, by = "ID")

# Display the transformed data and merged data
print("Transformed Data (Sample):")
head(data_transformed)

print("Merged Data:")
print(merged_data)
