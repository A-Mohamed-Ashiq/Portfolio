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

We see that the Corrected_Sentiment score ranges from -4 to 4. Thus, it is assumed that scores below 0 are negative sentiments, 0 is neutral and greater than 0 are positive sentiments.

The target variable Corrected_Sentiments was discretised into a flag variable with 0 representing negative and neutral sentiments and 1 representing positive sentiments. This allows better evaluation of the performance model as SPSS Modeler can only generate AUC-ROC reports for binary classification problems, which would be vital to us in dealing with heavily imbalanced dataset with majority of the tweets being neutral.

Python File to get discretised data and distribution of scores after correction: Click [here](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/blob/main/Python%20Codes/File%20to%20Get%20Corrected%20and%20Descritised%20Total%20Sentiments.ipynb).

After distretisation:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/04a55fde-45fd-4021-a269-2400b09f692c)



### Data Exploration:

Python File for Data Exploration: Click [here](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/blob/main/Python%20Codes/Data%20Exploration.ipynb).

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

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/259d7f0c-f3fd-4d29-9610-831cb4451f91)

Fig. 6 Display tweets posted during non-business hours vs during business hours



#### Using default CART on dataset
Feature selection: The inputs (X) that we will use will be State, retweet_count, reply_count, like_count, quote_count, Buzz, OpnHours, Day, Time, 4-phase, Followers, Vividness, ContentType, WC, Clout, Cognition, Affect, i and emotion. The target variable(Y) will be Corrected_Sentiment_Flagged.

Result:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/9fad3c2a-faf9-4727-9672-c3b4648e99b1)

At first glance, the overall testing accuracy is 84.75%, which is considered very good. However, the model failed to make any event predictions, instead predicting everything as a non-event. This is very likely due to the class imbalance between neutral sentiments and positive sentiments. For example, we know from data exploration that the majority class, which is negative and neutral sentiments, makes up 84.36% of all observations in the dataset. This tracks with the accuracy score as shown, which is also 84%, meaning the model just predicted everything as a non-event. 
Accuracy is calculated as follows: 19409 + 0 / (19409 + 0 + 3597 + 0) = 0.8436.
The AUC-ROC score is also 0.5, indicating that the model performs no better than random guessing.
As such, this model is completely unusable in solving the proposed problem since it does not even attempt to predict positive tweets.



### New Model Design
1. **Imbalanced Learn’s RandomUndersampler**
To undersample the majority class to a 1:1 ratio. The idea here is to randomly remove samples from the majority class until the classes are balanced. (Kumar & Abdelaziz, 2020) This may be suitable as the dataset is large with over 23k records, which means there is not a significant loss of information. The majority class is also less important than the minority class in this case since we are only interested in predicting positive sentiments. Thus, undersampling will force the model to focus more on the minority class. (Kumar & Abdelaziz, 2020) However, because it removes records by random, it will cause data loss and possibly cause underfitting.

2. **Random Oversampling**.
Similar to the previous approach, this method involves randomly upsampling the minority class. However, by duplicating minority records randomly, there is a risk of overfitting as the machine might overly rely on duplicated examples during learning.

3. **IBM SPSS Modeler Balance Node**
This node can be used to balance data by duplicating and then discarding records based on the specified conditions (IBM, n.d.). Using it to both oversample and undersample the minority and majority classes respectively, which might lead to better performance.

4. **Synthetic Minority Over-sampling Technique (SMOTE) Node in SPSS Modeler**
SMOTE serves as a preprocessing technique to tackle class imbalance within the dataset. It aims to mitigate overfitting, a typical issue associated with upsampling the minority class, as it prevents the machine from relying excessively on duplicated examples during the learning process. This is done by creating new synthetic samples that are close to the other minority observations instead of just randomly duplicating the minority class examples. (Maklin, 2022) However, SMOTE typically does not work well on datasets with categorical features even after encoding, which is the case in our dataset. This is due to the nature of the algorithm, which uses interpolation. (Kumar & Abdelaziz, 2020) Given that SPSS Modeler offers a dedicated SMOTE node, we will explore its utility in our specific context.

5. **SMOTE-NC**
Potentially the best method for our dataset. Synthetic Minority Oversampling Technique-Nominal Continuous is designed to be applied to mixed feature datasets like ours. In fact, it will not work unless it has both categorical and numerical features. The algorithm is similar to SMOTE for continuous variables, but now the nominal features will be considered using the value most common among the KNNs. (Kumar & Abdelaziz, 2020).

6. **Logistic Regression**
Using LR instead of a decision tree as it tends to be less vulnerable to imbalanced data, when compared to decision trees.


Finally, as we are dealing with a heavily imbalanced dataset, we need to use a different evaluation metric than accuracy. Thus, for the rest of the models, we will be using the **Sensitivity** of the model (how good the model is at picking up events), the **hit-rate for events, also known as precision** (How good the model is at event guesses) and **AUC-ROC**. For our new model we strive for a model with sensitivity above 60%, precision above 50% and a acceptable AUC-ROC score, above 0.7.


#### 1. Using Random UnderSampler with Python
Click [here](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/blob/main/Python%20Codes/Random%20Under%20Sampling.ipynb) for the code. 

#### 2. Using SMOTE node in IBM SPSS Modeler
1.	Attach the SMOTE node to the partition node
2.	Use default parameters
3.	Attach Distribution Graph and compare with original distribution
Before:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/fae271bf-8670-49ce-b1d7-1d73dc8ee0c0)

After:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/6af53355-4e08-427f-939d-1fb0fa91fd37)

#### 3. Using Balance Node in IBM SPSS Modeler
1.	Attach the balance node to the partition node.
2.	Set the conditions and factors as follows:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/6af8859d-704a-4f8b-b6bf-48e4ddeeaf75)


3.	Attach distribution graph to the balance node and the partition node and compare the distribution before balancing and after:
Before:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/86d7bca7-a986-4579-a43e-af248673c43e)

After:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/d2e26d17-9fe4-46d6-b976-22147db70650)

#### 4. SMOTE_NC with CART in Python
Click [here](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/blob/main/Python%20Codes/SMOTE-NC%20CART%20.ipynb) for the code. 

#### 5. SMOTE_NC with Logistic Regression in Python
Click [here](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/blob/main/Python%20Codes/Logit%20with%20SMOTE-NC.ipynb) for the code. 



### New Model Construction
#### New Default CART Model after Balancing data:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/359059ff-e60f-4f66-bd90-8138bcfb076d)

#### New Cart Model After Random UnderSampling with max depth 4:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/63b9ac6f-f818-4bdc-ac10-2fcfccbcbf5a)

#### New Cart Model After Random OverSampling with max depth 4:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/5ac66278-a538-4471-b93b-6a9ed604d11f)

#### New Default CART Model after SMOTE:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/38594a60-92fa-4d0a-9fd9-5cfbb7c591bf)

#### New Default CART Model after SMOTE-NC with max depth 4:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/1fed0de3-e11a-483f-bd1b-cf67dba348ae)



### Analysis and Interpretation
#### RandomUndersampling:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/e09f7dcd-3af1-418c-a3ac-80b6f4bbd91d)

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/20bf2784-041b-4b71-8d69-ca76a4731987)

Training and testing accuracy, 68.18% and 64.72 do not differ by much, suggesting that no/slight overfitting issues. While sensitivity is good (71.12%), Precision (26.79%) and AUC-ROC (0.67) are poor. It seems that the resampling method is ineffective in addressing the class imbalance. However, this serves a good starting point to compare further models on.

#### Random Oversampling:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/5c2912de-9b2a-42c2-a0d1-613bc57b5164)

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/2d108288-6128-46e2-b8f2-5e5758a02cf8)

Training and test scores are 67.13% and 63.5% respectively. This suggests that there is no/slight overfitting issues. Hit-rate for event is 25.32%. Sensitivity is 69.44%. AUC-ROC is 0.66. The testing and training accuracy do not differ by much, again suggesting no issues with overfitting. Overall performance is comparable to the random undersampling model, but slightly worse.

#### Balance node (Mix of both Under and Over Sampling)

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/137ebe21-c732-48bf-9c23-6ddd690411e6)

Overall Training and Testing accuracy is 70.12% and 70.84% respectively, suggesting no issues with overfitting. Sensitivity of the Model is good, with a sensitivity of 64.9%. Hit rate for “Event” is poor at 29.36%. The ROC-AUC is 0.745, which is considered on the lower end of acceptable. Again, comparable performance to all the models thus far.

#### SMOTE node:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/fb8bdc3c-b570-482b-8c5c-11f4cbea6139)

Overall training and testing accuracy of the model is 72.71% and 72.45% respectively, suggesting that the model is not overfitted. Sensitivity is 60.38%, which is below average. Precision is 31.07%, which is poor. The AUC-ROC is 0.73, which is acceptable. Here we see the model's precision improved by sacrificing its sensitivity.

#### SMOTE-NC:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/5f293378-4a08-4810-94d2-91c2d371a54f)

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/83da79a0-9497-41a8-bf37-8d75b011205e)

Training and testing accuracy, 70.76% and 72.23% do not differ much, although there might be some underfitting issues as the test accuracy is higher than the training accuracy. Precision is poor, 29.18%, Sensitivity is also the poorest out of all the models thus far (55.42%). The AUC-ROC score (0.65) is also poor.

#### SMOTE-NC + Logistic Regression:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/b4eca733-924e-495e-851b-c1d823d5054a)

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/18283a1a-ee1b-4415-9d8c-5acd7164a32a)

The training and testing accuracy (71.4% and 68.21%) does not differ by much, suggesting no overfitting. Sensitivity(64.11%) is acceptable, while precision (27.48%) and AUC-ROC (0.67) is still poor. Despite logistic regressions supposedly being less prone to imbalanced data, its performance here is comparable to CART. This may be due to how a single linear equation, log odds, may not be able to accurately split the complex, non-linear relationship for social media data, unlike a decision tree, which is more robust in fitting non-linear relationships.

### Compare and Conclude
All the new models are far superior to the baseline model, considering that it can predict events with good sensitivity (except SMOTE-NC CART), instead of only making non-event predictions. We decided that the best model for our business problem would be using a mix of both undersampling and oversampling (the balance node), since it has good sensitivity (64.9%) which is above 60% and the second highest precision scores (29.36%.) out of all the models. It is also one of the few models with acceptable AUC-ROC score (0.745).
Nevertheless, the hit rate for events, which is 20.64% worse than flipping a coin, leaves a lot to be desired even if there is very little consequence for misclassification in our business problem. This all means that even though the model is good at picking up events, it is extremely prone to giving false positives. Ideally, the precision should be at least above 50%.

Our goal is to build a predictive model for DMOs using tweet content to guide social media strategies during lockdown phases. Despite addressing significant class imbalance with resampling methods, our models achieved modest sensitivity and precision. The most effective approach involved a combination of random oversampling and undersampling. Surprisingly, SMOTE-NC, although technically advanced, yielded poor results. Nonetheless, these models remain valuable for informing strategies.
From the “best model” with the IBM balance node, we can derive Social Media Strategies:

![image](https://github.com/A-Mohamed-Ashiq/Portfolio-Projects/assets/104308123/b5fd8516-c154-47f0-b706-7e1127666ba0)

For example, the model predicts that if the DMOs tweet posts that has less than 45 words, promoting Assam, HP, J and K, UP and Uttarkhand and is a text-based tweet, then 78.83% of such tweets will result in positive sentiments. Another example: tweets with less than 45 words, promoting Bihar, Delhi, Karnataka, Rajasthan, Telangana and West Bangal during the pre-covid phase with an emotion score of less than or equal to 2.53 will result in positive sentiments 64.7% of the time. 

**Appraisal of Limitations**

Due to the low precision of the model, DMOs relying on these predictions and splitting rules must consider the possibility of false positives and use caution in implementing strategies derived from the models. Conducting further market research and focus groups will be necessary to help confirm leads. 
Since resampling did not fully address the issue, it may indicate insufficient representation of the minority class, suggesting the need for more data collection. Additionally, it suggests that the input features used in our model may not sufficiently capture the relevant information needed for distinguishing between positive and neutral/negative sentiments and may require more relevant features to be identified, collected and used instead. 






--------------------------------------------------------------------------------------
References:
Abhishek, K., & Abdelaziz, M. (2023). Machine Learning for Imbalanced Data. Packt Publishing. https://learning.oreilly.com/library/view/machine-learning-for/9781801070836/

IBM. (n.d.). Balance node In IBM Watson Knowledge Catalog as a Service. (SPSS Modeler). Retrieved march, 2024 from https://www.ibm.com/docs/en/watsonx-as-a-service?topic=operations-balance-node

Li, S. (2017, September 29). Building a logistic regression in Python, step by step. Medium. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8 

Maklin, C. (2022, May 15). Synthetic minority over-sampling technique (smote). Medium.       https://medium.com/@corymaklin/synthetic-minority-over-sampling-technique-smote-7d419696b88c 

