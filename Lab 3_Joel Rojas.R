---
  title: "Lab 3"
font-family: 'Corbel'
output: github_document
---
install.packages("tidyverse")
install.packages("class")
install.packages("caret")
library(tidyverse)
library(class)
library(caret)

  
  
  
  ### Econ B2000, MA Econometrics
  ### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY
  ### Fall 2023
  
  For this lab, we will use simple k-nn techniques of machine learning to try to guess people's plans to vaccinate their kids. Knn is a fancy name for a really simple procedure:

* take an unclassified observation
* look for classified observations near it
* guess that it is like its neighbors

We can understand the k-nn method without any statistics more complicated than means (of subgroups) and standard deviations. I posted a video. It's called "k-nn" since it uses the k Nearest Neighbors, where k is usually a small number, for instance the 3 nearest neighbors would be setting k=3.

We will compare this k-nn technique with a simple OLS regression.

Split into groups. You get the 75 min to discuss and then chat at the end of class (as usual, then write up results in homework). 

The idea here is to try to classify people's plans to vaccinate their kids. You probably have some thoughts about what is important in that classification. Here we try to train the computer, using the HHPulse data again.

Start with looking at the differences in means of some of the variables and put that together with your own knowledge of the world. You will obviously subset the data into those with kids. We'll start with the group with kids 12-17 years old. We'll classify based on education and what state the person lives in. (Not making a statement of causation! Think about why causal arrow can go in either direction.) Some of your previous work should come in handy.

Then use a k-nn classification.

You might experiment with other classifications whether the other kids age ranges or other aspects.

```{r eval = TRUE, echo=TRUE, warning=FALSE, message=FALSE}
setwd("..\\HouseholdPulse_W57")
load("Household_Pulse_data_w57.RData")
setwd("..\\ecob2000_lab3")
load("Household_Pulse_data_w57.RData")
# note your directory structures might be different so "setwd" would be different on your machine
require(tidyverse)
require(class)
require(caret)

```


```{r eval = TRUE}
summary(Household_Pulse_data$KIDGETVAC_12_17Y)
```



Restrict to those who have kids in that age:
```{r eval = TRUE}


dat_kidvaxx_nonmissing <- subset(Household_Pulse_data, (Household_Pulse_data$KIDGETVAC_12_17Y != "NA") )

```

```{r eval=FALSE}
# always check to make sure this went right!
summary(dat_kidvaxx_nonmissing)
```

See that some households have kids in other age ranges, you should look at how their other decisions about vaxx are related.

Next we'll transform these from a series of answers such as "definitely get" or "definitely NOT get" into a numeric scale.

We'll transform "definitely yes" to be 5, "probably yes" to be 4, "probably no" to 2, and "definitely no" to 1. Then both "unsure" and "do not know plans" are coded as 3.

This is obviously wrong! But might be useful. Why is it wrong? Because when we use numbers, there's a distance -- between 4 and 5, between 1 and 2. In converting to numbers, we're implicitly assuming those distances between answers are the same. But the distance (in the sense of how much would persuade a person) is likely different. The amount of information to get a person from "probably" to "unsure" is pretty small. While the amount of info to get away from "definitely" is larger. We will play around to see if this recode is useful, though. That's why you see it done often, out in the world.

Here's the code, with the summary statements so you can examine each step.

```{r eval = TRUE}

temp1 <- fct_recode(dat_kidvaxx_nonmissing$KIDGETVAC_12_17Y, '5' = 'kids 12-17yo definitely get vaxx',
                    '4'='kids 12-17yo probably get vaxx', '3'='unsure kids 12-17yo get vaxx',
                    '2'='kids 12-17yo probably NOT get vaxx', '1'='kids 12-17yo definitely NOT get vaxx',
                    '3'='do not know plans for vaxx for kids 12-17yo')
summary(temp1)

# this converts factor to numeric, note the odd syntax,
kidsvax1217 <- as.numeric(levels(temp1))[temp1]
summary(kidsvax1217)

```


What variables do you think are relevant in classifying whether somebody will get their kid(s) vaxxed against Covid? I'll try education and State. You should find other data to try to do better.

Here is some code to get you started. 

```{r eval = TRUE}
norm_varb <- function(X_in) {
  (X_in - min(X_in, na.rm = TRUE))/( max(X_in, na.rm = TRUE) - min(X_in, na.rm = TRUE)  )
}

# this is a lazier way ot converting factors to numbers, uses the order of the levels
data_use_prelim <- data.frame(norm_varb(as.numeric(dat_kidvaxx_nonmissing$EEDUC)),norm_varb(as.numeric(dat_kidvaxx_nonmissing$EST_ST)))

good_obs_data_use <- complete.cases(data_use_prelim,dat_kidvaxx_nonmissing$KIDGETVAC_12_17Y)
dat_use <- subset(data_use_prelim,good_obs_data_use)
y_use <- subset(kidsvax1217,good_obs_data_use)
```

Note that knn doesn't like factors, it wants numbers as inputs. So the "State" number is also stupid; you can do better.

Note that we often **normalize** the input variables and you should be able to convince yourself that a formula,
$$
(X - X_{min})/(X_{max} - X_{min})
$$
does indeed output only numbers between zero and one.

Next split the data into 2 parts: one part to train the algo, then the other part to test how well it works for new data. Here we use an 80/20 split.
```{r eval = TRUE}
set.seed(12345)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.8)
train_data <- subset(dat_use,select1)
test_data <- subset(dat_use,(!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]
```


Finally run the k-nn algo and compare against the simple means,
```{r eval = FALSE}
summary(cl_data)
summary(train_data)

for (indx in seq(1, 9, by= 2)) {
 pred_y <- knn3Train(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
 num_correct_labels <- sum(pred_y == true_data)
 correct_rate <- num_correct_labels/length(true_data)
 print(c(indx,correct_rate))
}
```

How can we compare this against another method, for instance a simple linear regression? Here's some code to get you started.

```{r eval = FALSE}
cl_data_n <- as.numeric(cl_data)
summary(as.factor(cl_data_n))
names(train_data) <- c("norm_educ","norm_state")


model_ols1 <- lm(cl_data_n ~ train_data$norm_educ + train_data$norm_state)

y_hat <- fitted.values(model_ols1)

mean(y_hat[cl_data_n == 2])
mean(y_hat[cl_data_n == 3])
mean(y_hat[cl_data_n == 4])
mean(y_hat[cl_data_n == 5])

# maybe try classifying one at a time with OLS

cl_data_n2 <- as.numeric(cl_data_n == 2) # now this is only 1 or 0, depending whether the condition is true or false

model_ols_v2 <- lm(cl_data_n2 ~ train_data$norm_educ + train_data$norm_state)
y_hat_v2 <- fitted.values(model_ols_v2)
mean(y_hat_v2[cl_data_n2 == 1])
mean(y_hat_v2[cl_data_n2 == 0])

# can you do better?

```


Now find some other data that will do a better job of classifying. How good can you get it to be? At what point do you think there might be a tradeoff between better classifying the training data and doing worse at classifying the test data?
  
  
  * A note on how "missing" values are coded. Note the difference here:
  ```{r eval=TRUE}

sum(is.na(Household_Pulse_data$KIDGETVAC_12_17Y))
sum(as.numeric(Household_Pulse_data$KIDGETVAC_12_17Y == "NA"))

```
There are costs and benefits to each approach. R is finicky about NA values in a way that Python is not. This particular variable has a "soft" NA value where that's just another factor level.
