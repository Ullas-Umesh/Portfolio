setwd("D://Stats and Visualization/Coursework")
data_file = read.csv("Health-Insurance-Dataset.csv")
attach(data_file)

#Viewing the dataset
head(data_file)
tail(data_file)
View(data_file)

#Checking for datatypes of each variable to findout continuous variables and categorical variables
data_file_dtypes = str(data_file)

#Checking for missing values
missing_values = colSums(is.na(data_file))
print(missing_values)

#Summarise the data
summary_data = summary(data_file)
print(summary_data)


#Evaluating standard devaition and variance
continuous_variables <- c("age", "bmi", "children", "charges")

apply(data_file[continuous_variables],2,sd)

apply(data_file[continuous_variables],2,var)


## Now we can check the kurtosis value by downloading the 'moments' package
install.packages("moments")
library(moments)
apply(data_file[continuous_variables],2,kurtosis)

#count of Categorical data
count_male = sum(sex == 'male')
count_male
count_female = sum(sex == 'female')
count_female
sex_count = count_male + count_female
print(sex_count)

count_smoker = sum(smoker == "yes")
count_smoker
count_nonsmoker = sum(smoker == 'no')
count_nonsmoker
smoker_count = count_smoker + count_nonsmoker
print(smoker_count)

count_northeast = sum(region == 'northeast')
count_northeast
count_northwest = sum(region == 'northwest')
count_northwest
count_southeast = sum(region == 'southeast')
count_southeast
count_southwest = sum(region == 'southwest')
count_southwest
region_count = count_northeast+count_northwest+count_southeast+count_southwest
print(region_count)

#Visualising the dataset
install.packages('dplyr')
library(dplyr)
ggplot(data_file, aes(age)) +
  geom_histogram(aes(y = ..density..), binwidth = 2, fill = "skyblue", color = "black") +
  geom_density(alpha = .2, fill = "purple", lwd = 0.8) +
  labs(title = "Distribution of Age with Histogram and Density Plot", x = "Age",y="Density")

ggplot(data_file, aes(bmi)) +
  geom_histogram(aes(y = ..density..), binwidth = 2, fill = "skyblue", color = "black") +
  geom_density(alpha = .2, fill = "purple", lwd = 0.8) +
  labs(title = "Distribution of BMI with Histogram and Density Plot", x = "BMI",y="Density")

ggplot(data_file, aes(children)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.5, fill = "skyblue", color = "black") +
  geom_density(alpha = .2, fill = "purple", lwd = 0.8) +
  labs(title = "Distribution of Children with Histogram and Density Plot", x = "Children",y="Density")

ggplot(data_file, aes(charges)) +
  geom_histogram(aes(y = ..density..), binwidth = 400, fill = "skyblue", color = "black") +
  geom_density(alpha = .2, fill = "purple", lwd = 0.8) +
  labs(title = "Distribution of Charges with Histogram and Density Plot", x = "Charges",y="Density")


# Bar plot for 'sex'

barplot(table(sex), main = "Distribution of Sex", xlab = "Sex", col = c("pink", "skyblue"))

# Bar plot for 'region'
barplot(table(data_file$region), main = "Distribution of Region", xlab = "Region", col = rainbow(length(unique(data_file$region))))

# Bar plot for 'smoker'
barplot(table(data_file$smoker), main = "Distribution of Smoker", xlab = "Smoker", col = c("lightgreen", "lightcoral"))

# Load the corrplot package
install.packages("corrplot")
library(corrplot)


# Calculate the correlation matrix
correlation_matrix <- cor(numeric_data)

# Visualize the correlation matrix
corrplot(correlation_matrix, method = "color", type = "lower", tl.cex = 0.7)





##################################################################################
#2nd Question
#Visualising the dataset using scatter plot
install.packages("psych")
library(psych)
numeric_data = data_file[,c("age",'bmi','children','charges')]
pairs(numeric_data)

#Test for normality
kstest_age = ks.test(age, "pnorm",mean = 39.21, sd = 14.04996)
print(kstest_age)

kstest_bmi = ks.test(bmi,"pnorm",mean = 30.66,sd = 6.09818)
print(kstest_bmi)

kstest_children = ks.test(children,"pnorm",mean = 1.095, sd= 1.2054)
print(kstest_children)

kstest_charges = ks.test(charges,"pnorm",mean = 13270, sd = 12110)
print(kstest_charges)

# Selecting continuous variables or Pearson's Correlation coefficients
install.packages("GGally")
library("GGally")
continuous_vars <- data_file %>% select(age, bmi, children, charges)
# Calculating correlation
cor_matrix <- cor(continuous_vars, use = "complete.obs")
print(cor_matrix)

age_charges_spearman = cor.test(age, charges, method = 'spearman')
print(age_charges_spearman)

age_bmi_spearman = cor.test(age, bmi, method = 'spearman')
print(age_bmi_spearman)

age_bmi_pearson = cor.test(age, bmi, method = 'pearson')
print(age_bmi_pearson)

age_children_spearman = cor.test(age, children, method = 'spearman')
print(age_children_spearman)

bmi_children_pearson = cor.test(bmi, children, method = 'pearson')
print(bmi_children_pearson)

bmi_children_spearman = cor.test(bmi, children, method = 'spearman')
print(bmi_children_spearman)

bmi_charges_pearson = cor.test(bmi, charges, method = 'pearson')
print(bmi_charges_pearson)

bmi_charges_spearman = cor.test(bmi, charges, method = 'spearman')
print(bmi_charges_spearman)

children_charges_spearman = cor.test(children, charges, method = 'spearman')
print(children_charges_spearman)

#Chi - Square test for categorical variables
sex_smoker_chisquare = chisq.test(sex,smoker)
print(sex_smoker_chisquare)

sex_region_chisquare = chisq.test(sex,region)
print(sex_region_chisquare)

smoker_region_chisquare = chisq.test(smoker,region)
print(smoker_region_chisquare)









#######################################################################################

#3rd Question
install.packages("fastDummies")
library(fastDummies)

# Create dummy variables
lm_data <- fastDummies::dummy_cols(data_file, select_columns = c("smoker", "sex", "region"))

# View the first few rows of the new dataframe to verify the dummy variables
View(lm_data )

# Fit the linear regression model
model <- lm(charges ~ age + sex_male + bmi + children + smoker_yes+region_northeast+region_northwest+region_southeast, data = lm_data)

# Output the summary of the model to view coefficients and statistics
summary(model)

#Visualising linear regression model
install.packages('car')
library(car)
crPlots(model,ylab = "Charges")








####################################################################################################################################

#4th Question
install.packages("dplyr")
library(dplyr)

data_category_A <- filter(data_file, charges <= median(charges))
data_category_B <- filter(data_file, charges > median(charges))
tail(data_category_A)
tail(data_category_B)

summary(data_category_A)
summary(data_category_B)

# Combining 'data_category_A' and 'data_category_B'
combined_data <- rbind(data_category_A, data_category_B)
View(combined_data)

#Test for normality - Category A
ks.test(data_category_A$age, "pnorm", mean(data_category_A$age), sd(data_category_A$age))
ks.test(data_category_A$bmi,"pnorm",mean(data_category_A$bmi),sd(data_category_A$bmi))
ks.test(data_category_A$children,"pnorm",mean(data_category_A$children), sd(data_category_A$children))
ks.test(data_category_A$charges,"pnorm",mean(data_category_A$charges), sd(data_category_A$charges))

#Test for normality - Category B
ks.test(data_category_B$age, "pnorm",mean(data_category_B$age), sd(data_category_B$age))
ks.test(data_category_B$bmi,"pnorm",mean(data_category_B$bmi),sd(data_category_B$bmi))
ks.test(data_category_B$children,"pnorm",mean(data_category_B$children), sd(data_category_B$children))
ks.test(data_category_B$charges,"pnorm",mean(data_category_B$charges), sd(data_category_B$charges))


#Test for differences
#Mann WHitney U test
# Assuming 'group1' and 'group2' are your two independent groups
result_age = wilcox.test(data_category_A$age, data_category_B$age, alternative = "two.sided")
result_age
result_bmi = t.test(data_category_A$bmi, data_category_B$bmi, alternative = "two.sided")
result_bmi
result_children = wilcox.test(data_category_A$children, data_category_B$children, alternative = "two.sided")
result_children
result_charge = wilcox.test(data_category_A$charges, data_category_B$charges, alternative = "two.sided")
result_charge
summary(data_category_A)

#Chi - square test
chisq.test(data_category_A$sex,data_category_B$sex)
chisq.test(data_category_A$region,data_category_B$region)

#Combining categorical data and Visualising the dataset
data_file$CHARGE_split = ifelse(data_file$charges <= median(charges), "Category A", "Category B")
View(data_file)


#Plot
# Box plot for 'age' by CHARGE-split
ggplot(data_file, aes(x = CHARGE_split, y = age, fill = CHARGE_split)) +
  geom_boxplot() +
  labs(title = "Box Plot of Age by CHARGE-split", x = "CHARGE-split", y = "Age")

# Box plot for 'bmi' by CHARGE-split
ggplot(data_file, aes(x = CHARGE_split, y = bmi, fill = CHARGE_split)) +
  geom_boxplot() +
  labs(title = "Box Plot of BMI by CHARGE-split", x = "CHARGE-split", y = "BMI")

# Box plot for 'children' by CHARGE-split
ggplot(data_file, aes(x = CHARGE_split, y = children, fill = CHARGE_split)) +
  geom_boxplot() +
  labs(title = "Box Plot of Children by CHARGE-split", x = "CHARGE-split", y = "Children")

# Box plot for 'charges' by CHARGE-split
ggplot(data_file, aes(x = CHARGE_split, y = charges, fill = CHARGE_split)) +
  geom_boxplot() +
  labs(title = "Box Plot of Charges by CHARGE-split", x = "CHARGE-split",y="Charges")







###################################################################################################################################

#5th Question
#Kruskal-Wallis rank sum test - age by region
kruskal_result_age = kruskal.test(age ~ region, data = data_file)
print(kruskal_result_age)

#ANOVA - bmi by region
anova_result_bmi = aov(bmi ~ region, data = data_file)
summary(anova_result_bmi)

#post hoc test
TukeyHSD(anova_result_bmi)

#Visualising results for posthoc test
par(mfrow = c(1, 1))
boxplot(bmi ~ region, data = data_file, col = "lightblue", main = "BMI Distribution by Region", ylab = "BMI", xlab = "Region")
boxplot(age ~ region, data = data_file, col = "lightgreen", main = "Age Distribution by Region", ylab = "Age", xlab = "Region")

#Kruskal-Wallis rank sum test - children by region
kruskal_result_children = kruskal.test(children ~ region, data = data_file)
print(kruskal_result_children)

#Chi-square test
sex_geography_chisq = chisq.test(sex,region)
print(sex_geography_chisq)

smoker_geography_chisq = chisq.test(smoker,region)
print(smoker_region_chisquare)
