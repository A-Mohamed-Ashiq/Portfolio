# Portfolio Projects 
## Project 1 - Data Mining
### Title: Sentiment Analysis on Social Media tweets for DMOs in India
#### Dataset
Download the dataset from Kaggle [here](https://www.kaggle.com/datasets/jocelyndumlao/dmo-social-media-engagement-dataset/)

#### Tools
- Python
- IBM SPSS Modeler

#### Background
The global tourism industry, including Destination Management Organisations (DMOs), has undergone profound changes due to the recent COVID-19 pandemic. Throughout various stages of lockdown and even in the post-lockdown era, there have been notable shifts in practices, customer behaviours, and expectations across the industry. Many organisations failed to adapt in time, and tourism suffered.

#### Motivation
Considering possible new lockdowns from a resurgence of covid 19 cases or from new diseases, DMOs must examine and comprehend these emerging trends and adaptations thoroughly. By doing so, they can effectively strategize and implement measures to enhance their business performance and ensure long-term success should a new lockdown be imposed.

#### Business Problem 
The business problem is that DMOs cannot identify content strategies or upload schedules (I.e. what kinds of tweets or during business hours or not) that will result in positive sentiments during different phases of the pandemic. Thus, the business objective is to accurately predict if a tweet can generate strong sentiments in social media users that react to their social media content strategy during the different phases.

#### Objective
The data mining objective will be to predict if a particular post can invoke positive sentiments in social media users using predictive models like **logistic regressions** and **decision trees** etc, and then analyse the splitting criteria to better understand what can invoke positive sentiments in users and help inform strategies. 

#### Data Quality and Cleaning
Data size: There are 23,006 records with 41 fields in this dataset. There are no missing values.
There is an error in the calculation of the total sentiment values and corrected this error by adding “Positive” and “Negative” values to Corrected_Sentiment.

Error Found:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/d54a1e71-773a-426b-9773-1a1c7fa627a6)

Adding Corrected_Sentiment Variable:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/57a2a678-1eba-4fe7-97cb-c04df059701b)

Distribution of sentiment scores after correction:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/91e80dee-abd0-4062-b70e-f3c49da6af6a)

We see that the Corrected_Sentiment score ranges from -4 to 4. Thus, we assumed scores below 0 are negative sentiments, 0 is neutral and greater than 0 are positive sentiments.

We discretise the target variable Corrected_Sentiments into a flag variable with 0 representing negative and neutral sentiments and 1 representing positive sentiments. This allows better evaluation of the performance model as SPSS Modeler can only generate AUC-ROC reports for binary classification problems, which would be vital to us in dealing with heavily imbalanced dataset with majority of the tweets being neutral.

Python File to get discretised data and distribution of scores after correction: Click [here](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/blob/main/Python%20Codes/File%20to%20Get%20Corrected%20and%20Descritised%20Total%20Sentiments.ipynb).

After distretisation:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/04a55fde-45fd-4021-a269-2400b09f692c)


#### Data Exploration:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/324b1bd6-29a4-473a-8334-d09d1caf8df8)

Fig. 1 Displays the states most promoted by DMOs

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/f8f8d5cb-2977-4764-a122-8e14caeed339)

Fig. 2 Display Tweet distribution by Day of the Week

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/7edda3d6-3b1a-4382-ad18-668b4f9d9a6f)

Fig. 3 Display the type of tweet activity

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/3e09a23c-8e5a-4766-8528-8ff8943e7c21)

Fig. 4 Display the Tweet distribution between Business & Non-Business hours

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/0f6a3b3c-1c1f-4da3-a01f-ecd98f6c695a)

Fig. 5 Display shift in tweet content during non-business hours and business hours 

image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/259d7f0c-f3fd-4d29-9610-831cb4451f91)

Fig. 6 Display tweets posted during non-business hours vs during business hours



#### Using default CART on dataset
Feature selection: The inputs (X) that we will use will be State, retweet_count, reply_count, like_count, quote_count, Buzz, OpnHours, Day, Time, 4-phase, Followers, Vividness, ContentType, WC, Clout, Cognition, Affect, i and emotion. The target variable(Y) will be Corrected_Sentiment_Flagged.

Result:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/9fad3c2a-faf9-4727-9672-c3b4648e99b1)

At first glance, the overall testing accuracy is 84.75%, which is considered very good. However, the model failed to make any event predictions, instead predicting everything as a non-event. This is very likely due to the class imbalance between neutral sentiments and positive sentiments. For example, we know from data exploration that the majority class, which is negative and neutral sentiments, makes up 84.36% of all observations in the dataset. This tracks with the accuracy score as shown, which is also 84%, meaning the model just predicted everything as a non-event. 
Accuracy is calculated as follows: 19409 + 0 / (19409 + 0 + 3597 + 0) = 0.8436.
The AUC-ROC score is also 0.5, indicating that the model performs no better than random guessing.
As such, this model is completely unusable in solving the proposed problem since it does not even attempt to predict positive tweets.

#### 

















