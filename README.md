# AI_AutomaticPrediction-Classification

## 1)Goal : Given a number of observed data and their class labels(training data), we develop an algorithm to predict a class label of a new patten

## 2)개발언어 : Scikit-Learn and Anaconda, py

## 3)DataSets
-Ames_Housing_Sales : SalePrice 예측
-Human_Activity_Recongnition : Activity 예측
-Human_Resources_Employee_Attribution : Salary 예측
-Iris : Species 예측
-Orange_Telecom_Churn_Data : Churned 유무 예측
-Wine_Quality_Data : Color 예측

# 알고리즘 선택 

## 1)Ames_Housing_Sales
String data 제거
GradientBoostingRegressor사용 Parameter 최적화

## 2)Iris
MinMaxScaler사용, LinearDiscriminantAnalysis사용 Parameter최적화

## 3)Wine
StandardScaler사용, SVC사용 Parameter 최적화

## 4)Human_Activity_Recongnition
VotingClassifier사용
(1)먼저 다양한 모델들 중 가장 결과가 잘나오는 모델은 선택후 해당 모델이 틀리는 데이터를 간추렸다.
(2)이렇게 간추린 데이터를 바탕으로 다양한 모델들 중 가장 결과가 잘나오는 모델이 틀리는 것을 가장 잘맞추는 모델을 2개 선택
Voting을 이용해서 가장 잘나오는 모델을 보완하도록 설계

## 5)Human_Resources_Employee_Attribution
'department'특징에 LabelEncoder사용 VotingClassifier사용
(1)먼저 다양한 모델들 중 가장 결과가 잘나오는 모델은 선택후 해당 모델이 틀리는 데이터를 간추렸다.
(2)이렇게 간추린 데이터를 바탕으로 다양한 모델들 중 가장 결과가 잘나오는 모델이 틀리는 것을 가장 잘맞추는 모델을 3개 선택
(3)Employee에 경우 가장잘나오는 모델도 0.6 정도에 정확도를 가지므로 이 가장 잘 나오는 경우를 유지하기 위해 Voting 모드를 변경해 비율을 [2,1,1,1]의 비율로 Voting하도록 설계

## 6)Orange_Telecom_Churn_Data
VotingClassifier사용
(1)먼저 다양한 모델들 중 가장 결과가 잘나오는 모델은 선택후 해당 모델이 틀리는 데이터를 간추렸다.
(2)이렇게 간추린 데이터를 바탕으로 다양한 모델들 중 가장 결과가 잘나오는 모델이 틀리는 것을 가장 잘맞추는 모델을 2개 선택
Voting을 이용해서 가장 잘나오는 모델을 보완하도록 설계
