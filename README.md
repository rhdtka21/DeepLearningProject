# 

# ** AI+X Deep Learning Project **

[![img](https://1.bp.blogspot.com/-MnOCX8sGAJY/Xe4Fjhp-9mI/AAAAAAAAAE8/4mIZqP3ixG0SAhV1--hzwWga287Yxpx6gCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B5.27.23.png)](https://1.bp.blogspot.com/-MnOCX8sGAJY/Xe4Fjhp-9mI/AAAAAAAAAE8/4mIZqP3ixG0SAhV1--hzwWga287Yxpx6gCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B5.27.23.png)

### 

### [![img](https://4.bp.blogspot.com/-sLS1M-HL1vQ/Xe4NEAKPR0I/AAAAAAAAAFI/7bcCeKf3-qwlTgCCDIUh0isZoBqnCzkjwCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B5.59.55.png)](https://4.bp.blogspot.com/-sLS1M-HL1vQ/Xe4NEAKPR0I/AAAAAAAAAFI/7bcCeKf3-qwlTgCCDIUh0isZoBqnCzkjwCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B5.59.55.png)

## Members
### 융합전자공학부 2015003209 석정우, rhdtka21@naver.com

### - 데이터셋 프로세싱, 그래프 분석, 유튜브 촬영

### 컴퓨터소프트웨어학부 2018009116 정태현, wxogus25@gmail.com  

### - 로지스틱 회귀 구현, 그래프 분석, 블로그 글 작성 

### 융합전자공학부 2015003945 황재석, dajasin245@naver.com

### - 데이터셋 프로세싱, 그래프 분석, 블로그 글 작성

### 융합전자공학부 2015003127 배지환, ekone8@naver.com

### - XGBoost 알고리즘 적용, 그래프 분석, 블로그 글 작성





### I. Introduction

 본 프로젝트에서 설명할 내용은 Kaggle 사이트에 올라온 House Prices: Advanced Regression Techniques (https://www.kaggle.com/c/house-prices-advanced-regression-techniques)을 수행해 내는 것입니다. 이 프로젝트의 내용을 간략히 요약하자면 주택에 관한 여러가지 데이터들, 주택의 위치나 주택의 가격 등 주택의 특징을 분석하고 학습하여 주택의 특징으로 주택의 가격을 도출해 내는 것입니다.



### II. Datasets

 Dataset을 보면 단순히 연속적인 값들만 존재하는 것이 아니라 문자열 값들도 주어집니다. 그렇기 때문에 선형 회귀를 적용하는 것은 적합하지 않아서 위와 같은 모델들을 선정하였습니다. 또한, 여러 가지 feature가 존재하는 만큼 이 중에서 주택 가격 예측에 좋은 영향을 끼치는 feature도 있을 것이고, 나쁜 영향을 끼치는 feature도 있을 것입니다. 이러한 feature들의 영향력은 모델링 도중 가중치에 따라서 늘어나거나 줄어들 수 있으며, 차원 축소(PCA 알고리즘 등) 등의 방식을 통해 feature를 인위적으로 가공하거나 제거할 수도 있습니다.





### III. Methodology

- 이번 프로젝트의 과정에서는 2개의 알고리즘 방법인 XGBoost와 Logistic Regression을 각각 이용하여 이 두개의 알고리즘을 더욱 깊게 이해하면서 이 방법들이 어떠한 차이를 보이는지와 결과값을 도출해 내는 과정을 그래프를 통해 확인할 것입니다. 더욱 나아가 프로그래밍을 통해 딥러닝의 학습시키는 과정에 대해 이해하는 것이 프로젝트의 최종 목표입니다.

- 이 프로젝트에서 DataSet의 주어진 특징의 수는 81개입니다. 이것에 대해 간략하게 줄이자면 우선 주택의 등급으로 시작하여 주택의 위치(도로, 골목, 이웃, 편의시설 접근성 등), 주거환경, 주택 건설 또는 리모델링 연도, 주택의 재질(외벽 등), 집 내부(창고, 지하실, 수영장, 침실, 주방 등)에 관한 내용(얼마나 크며 무엇으로 이루어져 있고 그 안에 화장실 여부 등), 그리고 주택의 가격(월세, 전세 와 판매 방식)으로 이루어져 있습니다.

- 방법론적으로 보았을 때 위에 언급된 두 개의 알고리즘 XGBoost와 Logistic Regression은 데이터를 분석하여 주택의 가격을 예측하는데 사용되는 라이브러리, 모델입니다. 먼저, XGBoost는 Gradient Boosting 알고리즘을 병렬처리 등을 통해 최적화한 라이브러리입니다. Gradient Boosting 알고리즘은 앙상블 기법 중 Boosting에 속하는 기법으로 약한 분류기를 결합하여 강한 분류기를 만듭니다(https://3months.tistory.com/368). 다른 종류의 기법인 Bagging에 비해 성능이 더 좋지만 오버 피팅이 발생할 수 있다는 문제점이 있습니다(https://quantdare.com/what-is-the-difference-betweenbagging-and-boosting/). \

- XGBoost 라이브러리는 이러한 문제를 가지치기를 통해 해결한다는 장점이 있습니다(https://brunch.co.kr/@snobberys/137). 해당 알고리즘의 원리는 손실 함수를 최소화하여 정확한 모델을 만드는 것인데, 손실 함수를 파라미터로 미분해서 기울기를 구하고 값이 작아지는 방향으로 파라미터를 움직이면서 최소화 지점을 찾는 과정을 거칩니다. 미분한 값을 다음 분류기로 보내 피팅을 하고, 기존 모델은 이 새로운 모델을 흡수하여 손실함수를 줄이는 과정을 반복하면서 성능을 높입니다 (https://4four.us/article/2017/05/gradient-boosting-simply). 

- Logistic Regression은 기존의 선형 회귀로는 fitting하기 힘든 모델을 곡선으로 fitting하기 위해 사용하는 방식입니다. 로지스틱 함수(Logistic Function)와 승산(Odds)라는 개념을 차용하여 모델링하는데, 로지스틱 함수는 분야에 따라 시그모이드 함수라고 불리기도 합니다. 입력 값은 어떤 값이든 받을 수 있지만, 출력 결과는 항상 0에서 1 사이의 값을 가집니다. 이러한 확률밀도함수의 조건을 충족시키는 것을 로지스틱 함수라고 합니다. 승산은 임의의 사건 A가 발생하지 않을 확률 대비 일어날 확률의 비율을 뜻합니다. 즉, P(A)/1-P(A) 입니다. ([https://ratsgo.github.io/machine%20learning/2017/04/02/logistic/](https://ratsgo.github.io/machine learning/2017/04/02/logistic/)). 

- 두 모델은 모두 Python을 통해 데이터에 적용할 예정이며, 가능하면 적용하기 전, 혹은 이후 feature나 인자를 가공하고 변화를 주면서 알고리즘을 적용할 것입니다.



### IV. Evaluation & Analysis

- 아래는 실제로 코드를 통해 구현한 내용입니다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```



`matplotlib.pyplot` 는 Visualization을 위해 사용하는 패키지로 일반적인 차트나 플롯을 그리기 위해 불러왔습니다. `Seaborn` 또한 Visualization을 위해 사용하는 패키지로 통계용 차트를 그리기 위해 불러왔습니다. 각각 불러온 패키지 들은 사용하기 쉽도록 다시 재설정하였습니다.



------



```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
```



[![img](https://3.bp.blogspot.com/-f0sa1YmERCk/Xe3sp15IGqI/AAAAAAAAABA/NA1nQyOHmJoBT-ggKm5Bpp3liv5nqc06ACK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.41.38.png)](https://3.bp.blogspot.com/-f0sa1YmERCk/Xe3sp15IGqI/AAAAAAAAABA/NA1nQyOHmJoBT-ggKm5Bpp3liv5nqc06ACK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.41.38.png)



Train.csv와 test.csv파일을 불러 와서 train파일을 앞에서 5개의 데이터를 불러와 확인하는 작업으로 데이터가 정상적으로 불러왔는지 확인합니다.



------



```python
train.set_index('Id', inplace=True)
test.set_index('Id', inplace=True)
len_train = len(train)
len_test = len(test)
```

다음으로 train과 test에서 id열을 통해 데이터의 길이를 불러오고 저장하는 과정으로 원본 데이터를 변경하면서 불러옵니다.



------



```python
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>=0.2]
top_corr_features
```

[![img](https://4.bp.blogspot.com/-xQqZKxdd4eg/Xe3s5MqYFjI/AAAAAAAAABI/2NDyHLUK-4gVlrdq6yyjemRcGbj673FDwCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.42.38.png)](https://4.bp.blogspot.com/-xQqZKxdd4eg/Xe3s5MqYFjI/AAAAAAAAABI/2NDyHLUK-4gVlrdq6yyjemRcGbj673FDwCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.42.38.png)



다음으로 불러오는 데이터를 선택하는 과정입니다. 여기서 선택하는 데이터는 결과값과의 관계가 있는, 즉 다시 말해서 각각의 데이터를 가져와서 데이터 간의 관계가 서로 어느정도 상관관계가 있는 변수만을 가져와 사용합니다. 여기서 사용하는 상관관계의 값은 0.2이상인 데이터들을 사용합니다.



------



```python
plt.figure(figsize=(13,10))
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
```

[![img](https://4.bp.blogspot.com/-SrtoLPAFLLo/Xe3uoP5skKI/AAAAAAAAABY/ZUvsRiKWDAYxVXsi9BloZF4GVtTA06ogwCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.49.54.png)](https://4.bp.blogspot.com/-SrtoLPAFLLo/Xe3uoP5skKI/AAAAAAAAABY/ZUvsRiKWDAYxVXsi9BloZF4GVtTA06ogwCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.49.54.png)



여기서는 sns, 즉 처음 불러온 seaborn을 사용하여 데이터 간의 상관관계를 표로 시각적으로 표시하였으며 값이 커지면 초록색으로, 값이 작으면 빨간색으로 표현하여 값을 더 직관적으로 확인을 할 수 있게 하였습니다. 여기서 나타난 데이터 값은 위에서 얻어낸 데이터들 간의 상관관계가 0.2인 값 이상인 데이터들만을 가져온 것이며 아래에 보이는 표는 그 데이터들 간의 상관관계 값을 나타낸 것입니다.



------



```python
train_y_label = train['SalePrice']
train.drop(['SalePrice'], axis=1, inplace=True)
```

다음으로 우리가 구할 값인 SalePrice, 즉 주택 가격의 값을 미리 train 파일에서 미리 분리하였습니다.

첫 줄은 그 SalePrice 값을 다른 변수에 저장하였고 2번쨰줄에서는 SalePrice 값을 train에서 제거하여 변경시키는 과정입니다.



------



```python
house = pd.concat((train, test), axis=0)
house_index = house.index
print('Length of House Dataset : ',len(house))
house.head()
```

[![img](https://4.bp.blogspot.com/-DogkIcIhLpc/Xe3u8ea7heI/AAAAAAAAABk/DbBlp2M_-RUhdoYzQYaLI83IZRc5RaUUACK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.51.23.png)](https://4.bp.blogspot.com/-DogkIcIhLpc/Xe3u8ea7heI/AAAAAAAAABk/DbBlp2M_-RUhdoYzQYaLI83IZRc5RaUUACK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.51.23.png)


이제 train과 test를 합쳐서 한 개의 데이터 세트로 만들어 train과 test를 동시에 분석하기 위해 house라는 변수로 합쳐서 저장을 시켰습니다. 또한 house의 index를 따로 저장시키고 house가 정상적으로 만들어 졌는지 확인하기 위해 head()를 사용하여 앞의 5개의 데이터를 출력하여 확인하고 house의 길이를 출력시켜 총 몇 개가 있는 지 확인을 하였습니다. 합쳐준 결과 house의 데이터의 개수는 2919개가 되었습니다.



------



```python
check_null = house.isna().sum() / len(house)
check_null[check_null >= 0.5]
```

[![img](https://1.bp.blogspot.com/-zrNu31UruQI/Xe3vFdQgICI/AAAAAAAAABs/LGQRzMGy2kslUyTRYNXN2VSuNIjdfD2RwCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.52.02.png)](https://1.bp.blogspot.com/-zrNu31UruQI/Xe3vFdQgICI/AAAAAAAAABs/LGQRzMGy2kslUyTRYNXN2VSuNIjdfD2RwCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.52.02.png)

다음으로는 데이터의 값 중에서 NaN의 비율, 즉 데이터가 어느정도 비어있는지 확인하는 과정을 거쳤습니다. 비어 있는 데이터의 수를 전체 길이로 나누어 준 값으로 만약 모든 데이터가 비어 있지 않으면, 즉 모두 차 있으면 0이며 데이터가 비어 있을수록 1에 가까워 집니다. 모두 비어 있는 경우에는 1이 됩니다. 위 사진의 결과 값은 비어 있는 데이터 값이 많은 데이터들로 0.5보다 큰 값으로 대부분 값이 비어 있는 데이터들입니다. 여기서 우리는 0.5보다 큰, 즉 반 이상이 값이 비어 있는 데이터의 경우 다른 데이터들과는 다르게 처리해야 하는데 보통은 평균값으로 처리하는 반면 여기서는 비어 있는 값들이 0.9에 가까워 단순히 무시하기로 하였습니다.



------



```python
remove_cols = check_null[check_null >= 0.5].keys()
house = house.drop(remove_cols, axis=1)
house.head()
```

[![img](https://4.bp.blogspot.com/-RSGYU3oYyqU/Xe3vPc7j28I/AAAAAAAAAB4/cevHYMauNMgXRsvxuqibBZ9EZJjhuKZyQCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.52.40.png)](https://4.bp.blogspot.com/-RSGYU3oYyqU/Xe3vPc7j28I/AAAAAAAAAB4/cevHYMauNMgXRsvxuqibBZ9EZJjhuKZyQCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.52.40.png)

여기서는 비어 있는 빈도가 50%이상인 경우 즉, check_null의 값이 0.5 이상인 데이터들을 제거하는 과정입니다. 위에서 보면 알 수 있듯이 첫 번째 줄은 0.5인 데이터 줄을 찾아 변수에 저장하고 그 변수에 맞는 값들을 house 에서 제거하는 과정을 2번째 줄에서 수행합니다. 마지막으로 데이터가 정상적으로 지워졌는지 확인하기 위해 house의 앞 5개의 데이터를 출력해 확인을 하였습니다. 기존의 에는 79개의 데이터 종류가 존재하였지만 지금은 위의 4가지 데이터 종류가 사라진 75개만이 존재하는 것을 확인할 수 있습니다.



------



```python
house_obj = house.select_dtypes(include='object')
house_num = house.select_dtypes(exclude='object')

print('Object type columns:\n',house_obj.columns)
print('--------------------------------')
print('Numeric type columns:\n',house_num.columns)
```

[![img](https://2.bp.blogspot.com/-F_jgmld_hiE/Xe3vX0obxXI/AAAAAAAAACE/BLWAQ-CwnO8C2uNpCMuw45ZdocFfPuASACK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.53.13.png)](https://2.bp.blogspot.com/-F_jgmld_hiE/Xe3vX0obxXI/AAAAAAAAACE/BLWAQ-CwnO8C2uNpCMuw45ZdocFfPuASACK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.53.13.png)



다음으로는 데이터가 숫자로 이루어져 있는지, 아니면 오브젝트, 즉 글자로 이루어져 있는지 확인하여 구분 시키는 과정을 거쳤습니다. 나중에 데이터를 분석을 하는데 있어서는 숫자 값을 사용하는 것이 좋으며 그것을 위해 오브젝트로 이루어져 있는 값들과 숫자 값을 따로 분리시켜 오브젝트 값을 다시 가중치로 변환시켰습니다. 위에서는 우선 house_obj에 오브젝트로 이루어진 데이터 종류들을 저장하고, house_num에는 숫자로 이루어진 데이터 종류들을 저장하였습니다. 각각의 데이터 종류가 어떤 분류로 저장되었는지 확인하기 위해 숫자 값으로 이루어진 데이터 종류와 오브젝트로 이루어진 데이터 종류를 나누어 출력하였습니다.



------



```python
house_dummy = pd.get_dummies(house_obj, drop_first=True)
house_dummy.index = house_index
house_dummy.head()
```


[![img](https://4.bp.blogspot.com/-uooNXNYkfXg/Xe3vdvRegCI/AAAAAAAAACM/q3YnK6rMWDInBCRXrJblk3P9NGa9Os0aQCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.53.38.png)](https://4.bp.blogspot.com/-uooNXNYkfXg/Xe3vdvRegCI/AAAAAAAAACM/q3YnK6rMWDInBCRXrJblk3P9NGa9Os0aQCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.53.38.png)



다음으로 이전에 말한 것과 같이 오브젝트로 이루어진 데이터 종류들을 가중치로 바꾸어 전환시키는 과정입니다. 그리고 데이터들이 정상적으로 변환되었는지 확인하기 위해 가중치로 변한시킨 값을 house_dummy로 저장시키고 그것을 앞 5개의 데이터를 출력하여 확인하였습니다.



------



```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
imputer.fit(house_num)
house_num_ = imputer.transform(house_num)
house_num = pd.DataFrame(house_num_, columns=house_num.columns, index=house_index)

house_num.head()
```

[![img](https://2.bp.blogspot.com/-yUcrKa3htUI/Xe3vmbEh39I/AAAAAAAAACU/rXGCVKN7ULQX7DpHhTpk1Nbsu56mGx8cQCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.54.11.png)](https://2.bp.blogspot.com/-yUcrKa3htUI/Xe3vmbEh39I/AAAAAAAAACU/rXGCVKN7ULQX7DpHhTpk1Nbsu56mGx8cQCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.54.11.png)



다음으로 데이터 값이 비어 있는 NaN을 처리하는 과정입니다. 데이터가 50%이상 비어 있는 데이터 종류는 제거하여 무시하였지만 여전히 데이터 값들이 없는 곳은 존재합니다. 즉 위에서 구한 check_null의 값은 0.5 이하지만 데이터의 값이 없는 곳이 존재할 수 있다는 것입니다. 그것은 여기에서는 평균값으로 대체해주기 위해 sklearn.impute에서 SimpleImputer를 불러와서 NaN값을 데이터의 평균값으로 대체하였습니다. 또한 그것이 정상적으로 이루어졌는지 확인을 하기 위해 다시 앞 5개의 데이터를 출력하여 확인하였습니다.



------



```python
house = pd.merge(house_dummy, house_num, left_index=True, right_index=True)

house.head()
```

[![img](https://1.bp.blogspot.com/-qOoJo339RE4/Xe3vrjonnQI/AAAAAAAAACg/nj0qN94nH6A9JO51REDSBvPvLbSOK8P9wCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.54.34.png)](https://1.bp.blogspot.com/-qOoJo339RE4/Xe3vrjonnQI/AAAAAAAAACg/nj0qN94nH6A9JO51REDSBvPvLbSOK8P9wCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.54.34.png)



다음으로 오브젝트 값을 가중치로 바꾼 house_dummy와 숫자로 되어 있으면서 비어 있는 값을 평균값으로 대체한 house_num을 house에 다시 한 개로 합치는 과정입니다. 여기서 데이터가 합쳐지면서 순서가 섞이지 않도록 각각의 index 값을 True로 주어 순서대로 저장되게 하였습니다. 이것은 pandas를 이용해 실행하였으며 결과값을 확인하기 위해 앞 5개의 데이터 값을 출력하여 확인하였습니다.



------



```python
train = house[:len_train]
test = house[len_train:]
train['SalePrice'] = train_y_label

print('train set length: ',len(train))
print('test set length: ',len(test))
```

[![img](https://2.bp.blogspot.com/-WcqYpqMM7WQ/Xe3vwQjRJJI/AAAAAAAAACs/h74qZHl7GjYwjM652irXW80vQXUkKNWKgCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.54.53.png)](https://2.bp.blogspot.com/-WcqYpqMM7WQ/Xe3vwQjRJJI/AAAAAAAAACs/h74qZHl7GjYwjM652irXW80vQXUkKNWKgCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.54.53.png)



우선 test 값과 train의 데이터 세트의 개수를 세어서 저장한 것이며 test에는 SalePrice 값이 없으며 이 프로젝트의 목표 중 하나인 결과 값을 예측하는 것입니다. train에서는 SalePrice 값이 존재하니까 데이터 세트의 개수는 1개 차이가 됩니다.



------



```python
from sklearn.model_selection import train_test_split

X_train = train.drop(['SalePrice'], axis=1)
y_train = train['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

X_test = test
test_id_idx = test.index

print('X_train : ',len(X_train))
print('X_val : ',len(X_val))
print('X_test :',len(X_test))
```

[![img](https://4.bp.blogspot.com/-G8g1NLt02kg/Xe3v0DRkR_I/AAAAAAAAAC0/nnh8yiSzZUQ-94TDTfs1HCNg4emBSMWkgCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.55.08.png)](https://4.bp.blogspot.com/-G8g1NLt02kg/Xe3v0DRkR_I/AAAAAAAAAC0/nnh8yiSzZUQ-94TDTfs1HCNg4emBSMWkgCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.55.08.png)



그런데 여기서 test에서 SalePrice의 값을 예측한 뒤 그것을 검증하기 위한 값을 train에서 가져왔습니다. 우선 train에서 SalePrice 값 중 일부를 가져오기 위해 sklearn.model_selection에서 train_test_split를 가져와서 사용하였습니다. 여기서 검증값으로 사용하기 위해 train에서 가져온 SalePrice은 총 train의 SalePrice 개수의 20%만을 가져왔습니다. 여기서 가져온 값들은 순서대로 불러온 것이 아니라 Sheffle=True로 섞어서 무작위의 값을 가져온 것입니다. 또한 20%의 검증값을 X_val이라는 변수에 저장하였고 나머지는 X_train에 저장하였습니다. 이 값들이 정상적으로 저장되었는지 확인하기 위해 각각의 변수의 크기를 출력하여 확인하였습니다.



------



```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param = {
  'max_depth':[2,3,4],
  'n_estimators':range(550,700,50),
  'colsample_bytree':[0.5,0.7,1],
  'colsample_bylevel':[0.5,0.7,1],
}

model = xgb.XGBRegressor()

grid_search = GridSearchCV(
    estimator=model, param_grid=param, cv=5, 
    scoring='neg_mean_squared_error', n_jobs=-1, iid = False
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_estimator_)
```



[![img](https://1.bp.blogspot.com/-u7hjhoQx510/Xe3v9_QaqFI/AAAAAAAAAC8/zyoTweqydTcC3V0nozUVaUKLTqtro6PTACK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.55.44.png)](https://1.bp.blogspot.com/-u7hjhoQx510/Xe3v9_QaqFI/AAAAAAAAAC8/zyoTweqydTcC3V0nozUVaUKLTqtro6PTACK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.55.44.png)



다음으로는 이 프로젝트의 핵심인 XGBoost를 사용하여 데이터를 분석하고 예측하였습니다. 

우선 XGBoost를 사용하기 위해서 xgboost를 불러와 xgb로 줄여서 사용할 수 있게 하였습니다. 

또한 여기서는 그리드 알고리즘을 사용하여서 sklearn.model_selection에서 GridSearchCV를 불러와서 사용하였습니다. 우선 여기서 파라미터 설정인 param을 살펴보아야 합니다. 

제일 처음으로 나오는 max_depth는 tree의 depth를 설정한 것으로 2,3,4로 설정하였습니다. 다음은 n_estimators인데 이것은 들어갈 트리의 개수로 여기서는 2,3,4로 설정을 하였습니다. 

다음은 colsample_bytree로 각각의 tree에 대하여 변수를 샘플링하는 비율을 설정한 것으로 여기서도 범위 0.5~0.7로 설정하였습니다. colsample_bylevel도 colsample_bytree와 비슷하지만 여기서는 트리의 레벨별로 변수를 샘플링하는 비율을 설정한 것입니다. 

이것 또한 colsample_bytree와 같은 범위를 넣었습니다. 다음은 모델에 대해 설정한 것으로 XGBRegressor로 설정한 뒤 이것으로 그리드 서치를 만들어 교차 검증을 사용하였습니다. GridSearchCV에서의 변수로 사용된 것들을 살펴보겠습니다. 

처음으로 보이는 것은 estimator로 모형을 넣는 것으로 xgb.XGBRegressor()로 설정한 모델을 넣었으며 param_grid는 그리드에서 사용할 변수들을 이전에 설정한 param으로 설정하였습니다. 

다음으로 cv는 교차검증을 위한 값으로 여기서는 5로 설정을 하였습니다. N_jobs는 사용할 프로세서의 개수를 설정하는 것으로 -1은 모든 프로세스를 사용합니다. 여기서는 빠르게 값을 얻기 위해 -1값을 넣었습니다. id값은 가중치를 넣는 값이지만 여기서는 True와 False값이 차이가 없으나 기본값인 False로 설정하였습니다. 또한 scoring은 예측값을 검증하기 위해서 사용하는 점수로 여기서는 'neg_mean_squared_error'로 설정을 하였습니다. 

다음으로 나온 grid_search.fit은 그리드 서치를 사용하여 자동으로 여러개의 모형을 생성하고 이것에 대한 최적의 변수들을 찾아줍니다. 아래에서 그것을 출력한 것으로 첫번째 print는 최적의 변수 값을 다음으로는 최고의 점수를 낸 모형을 print하였습니다.

다음은 똑같은 데이터를 Logistic Regression을 이용하여 구현해 보았습니다. Logistic Regression: 로지스틱 회귀(Logistic Regression)은 분류와 회귀 두 종류의 문제에 적용 가능한 모델로 선형 회귀(Linear Regression)를 기반으로 한 모델입니다. 기존의 선형 회귀는 종속변수가 범주형 변수(포괄적으로 표현하여 이산(discrete)적인 변수)인 경우 모델링이 힘든 단점이 있습니다. 

예를 들어 왼쪽 아래의 그래프와 같이 데이터의 정의역이 {0, 1} 두가지 뿐인 경우 선형 회귀로는 잘 표현할 수 없습니다. 때문에 오른쪽 아래와 같은 S-curve 형태를 따른 로지스틱 회귀 모델이 제안되었습니다.

[![img](https://4.bp.blogspot.com/-_Ou9WNeLnWI/Xe3980ElN5I/AAAAAAAAADo/LRE8q4g1iI0boKWDMv1fr75KDfjyWlAzgCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.57.11.png)](https://4.bp.blogspot.com/-_Ou9WNeLnWI/Xe3980ElN5I/AAAAAAAAADo/LRE8q4g1iI0boKWDMv1fr75KDfjyWlAzgCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.57.11.png)

로지스틱 회귀는 확률 모델의 일종으로 독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법입니다. 가장 기본적인 이항형 로지스틱 회귀(binomial logistic regression)는 독립 변수가 [-∞, ∞]의 어떤 숫자이던 상관없이 결과 값이 항상 [0, 1] 사이에 있도록 합니다. 해당 결과 값을 확률 p라고 두고 odds ratio(승산률)을 다음과 같이 정의합니다.

[![img](https://2.bp.blogspot.com/-JYv1SNwy-Js/Xe3-G8_TfTI/AAAAAAAAADw/KZ4LdOPs2LUYdQRnlUh-B6qYrC1mEL5vgCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B4.56.04.png)](https://2.bp.blogspot.com/-JYv1SNwy-Js/Xe3-G8_TfTI/AAAAAAAAADw/KZ4LdOPs2LUYdQRnlUh-B6qYrC1mEL5vgCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B4.56.04.png)

즉, odds ratio는 어떤 사건이 일어날 확률과 그 사건이 일어나지 않을 확률의 비로 정의됩니다. 여기서 odds ratio의 로그 값을 함수로 정의합니다.

[![img](https://2.bp.blogspot.com/-m7A0Scfxfk0/Xe3-VKTaDmI/AAAAAAAAAEM/i0cMhv-CE9IZwbUhUtcRqMOzWXXB8MiUgCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B4.57.04.png)](https://2.bp.blogspot.com/-m7A0Scfxfk0/Xe3-VKTaDmI/AAAAAAAAAEM/i0cMhv-CE9IZwbUhUtcRqMOzWXXB8MiUgCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B4.57.04.png)

p는 [0, 1] 이므로 함수 f(p)의 범위는 실수 전체가 됩니다. 해당 함수를 p에 대한 식으로 나타내면 아래와 같은 식이 됩니다.

[![img](https://1.bp.blogspot.com/-McEF5r4oMjs/Xe3-NYnYr4I/AAAAAAAAAEE/mdGmf-bcP1Ecv857WO8YU_sFRpjMTo1YwCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B4.56.28.png)](https://1.bp.blogspot.com/-McEF5r4oMjs/Xe3-NYnYr4I/AAAAAAAAAEE/mdGmf-bcP1Ecv857WO8YU_sFRpjMTo1YwCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B4.56.28.png)

함수 f(p)를 z라고 하고, z에 대한 함수로 위의 식을 표현하면 아래와 같은 식이 됩니다.

[![img](https://1.bp.blogspot.com/-GyKWdbEaNZQ/Xe3-a8NKz1I/AAAAAAAAAEc/IXLhAuZnaqo8XzLUsdOiGdwWLWEfvaenwCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B4.57.28.png)](https://1.bp.blogspot.com/-GyKWdbEaNZQ/Xe3-a8NKz1I/AAAAAAAAAEc/IXLhAuZnaqo8XzLUsdOiGdwWLWEfvaenwCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B4.57.28.png)

위의 함수를 sigmoid 함수라고 부릅니다.



로지스틱 회귀는 
$$
ŷ = w[0] * x[0] + w[1] * x[1] + … + w[p] * x[p] + b
$$
 꼴의 값을 sigmoid 함수에 대입하여 그 결과 값을 가지고 가중치(w[])를 어떻게 업데이트 할지 정하는 방식으로 학습을 합니다. 학습 방식은 다른 모델들과 마찬가지로 경사 하강법(Gradient descent)을 사용합니다. 일반적인 로지스틱 회귀는 결과 값이 특정한 구간내의 값을 가지도록 하며, 이러한 특성을 이용하여 분류 모델로도 사용할 수 있고, 회귀 모델로도 사용할 수 있습니다.

모델에 적용한 테스트 데이터의 코드는 XGBoost에서 사용된 코드를 사용했으며, 두 모델의 분석을 위해 데이터 가공 역시 동일한 방식으로 진행하였습니다.



------



```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=13, solver='lbfgs', C=1000., multi_class='multinomial')

log_reg.fit(X_train, y_train)
```



먼저, sklearn.linear_model에서 LogisticRegression을 import 한 뒤 LogisticRegression 함수를 실행합니다. LogisticRegression 함수에서 사용된 parameter들은 다음과 같은 역할을 합니다.

random_state: random 데이터 생성 시 seed number가 됩니다.

solver: 최적화에 사용할 알고리즘을 정합니다. ‘liblinear’, ‘newton-cg’, ‘sag’, ‘saga’, ‘lbfgs’를 지원하며, multi_class 처리와 L2 regularization 등 조건에 따라 모델에 적합하다고 생각되는 알고리즘인 ‘lbfgs’(LBFGS, Limited Memory BFGS)를 선택하였습니다.

C: 최적화 조건을 찾기 위해 사용되는 정규화의 파라미터의 역수입니다. 정규화 파라미터가 커질수록 정규화를 강하게 하며, overfitting(과적합, 샘플 데이터에 너무 적합하게 학습하여 조금의 예외에도 큰 차이를 보이는 경우)이 발생할 확률이 커집니다.

multi_class: 다항 로지스틱 회귀를 사용하는 경우 ‘multinomial’로 설정합니다.



### V. Conclusion: Discussion

XGBoost의 결과를 분석하기 위해서 실제 SalePrice의 값과 예측한 SalePrice의 차이를 수치로 나타내기 위해 각각의 데이터의 실제값과 예측값의 차이값을 모두 더하여 데이터의 수로 나눈 값을 계산하였습니다.



------



```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

pred_train = grid_search.predict(X_train)
pred_val = grid_search.predict(X_val)

print('train mae score: ', mean_absolute_error(y_train, pred_train))
print('val mae score:', mean_absolute_error(y_val, pred_val))
```

[![img](https://3.bp.blogspot.com/-FMMcfZnXNcQ/Xe3wGkQd9MI/AAAAAAAAADM/rFViEMsQG6cWRcoRh-LBFeRVTwEHMjDagCK4BGAYYCw/s320/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.56.22.png)](https://3.bp.blogspot.com/-FMMcfZnXNcQ/Xe3wGkQd9MI/AAAAAAAAADM/rFViEMsQG6cWRcoRh-LBFeRVTwEHMjDagCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.56.22.png)



 그것을 위해 sklearn.metric에서 mean_squared_error와 mean_absolute_error를 불러와서 사용하였습니다. 다음으로 이것을 눈으로 확인하기 위해 train 파일의 검증값의 계산값과 예측값의 계산값을 출력하였습니다. 여기서 예측값의 차이의 평균이 16650정도가 되는 것을 확인할 수 있었습니다. 값이 자칫 커 보일 수가 있으나 실제로는 SalePrice 자체가 집값으로 매우 크기 때문에 실제 비율을 생각하면 큰 값은 아닙니다.



------



```python
plt.figure(figsize=(17,7))

plt.plot(range(0, len(y_val)), y_val, 'o-', label='Validation Actual')
plt.plot(range(0, len(pred_val)), pred_val, '-', label='Validation Predict')

plt.title('Prediction of House Prices')
plt.ylabel('Prices')

plt.legend()
```

 



[![img](https://4.bp.blogspot.com/-504b8bJWQp4/Xe3wMyv1JeI/AAAAAAAAADU/YlrTjtl9bQohWAT_U9L8BPJCR3AuCd1RgCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B3.56.44.png)](https://4.bp.blogspot.com/-504b8bJWQp4/Xe3wMyv1JeI/AAAAAAAAADU/YlrTjtl9bQohWAT_U9L8BPJCR3AuCd1RgCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B3.56.44.png)

마지막으로 예측값과 실제값을 비교하기 위해 다시 그래프로 표현한 것으로 파란색은 실제값, 노란색은 예측값으로 나타냈습니다. 이 표에서 Y값은 SalePrice이며 x값은 단순한 데이터를 나열한 것입니다.



Logistic Regression 모델의 검증은 XGBoost와 마찬가지로 실제 가격과 예측 가격의 차이를 데이터의 수로 나누는 방법을 사용했습니다. 중간에 C를 여러 차례 바꾸면서 진행하였으나 유의미한 차이를 보이지는 않았습니다.



------



```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

pred_train = log_reg.predict(X_train)
pred_val = log_reg.predict(X_val)

print('train mae score: ', mean_absolute_error(y_train, pred_train))
print('val mae score:', mean_absolute_error(y_val, pred_val))
```



`train MAE score: 36456.42123287671`

`val MAE score: 38403.356164383564`



XGBoost와 달리 학습에 사용한 데이터의 오차와 테스트 데이터의 오차가 크게 다르지 않았습니다. XGBoost와 같은 데이터 셋을 사용했으므로, 데이터 셋의 문제가 아닌 해당 모델의 한계라고 생각됩니다.



------



```python
plt.figure(figsize=(17,7))

plt.plot(range(0, len(y_val)), y_val,'o-', label='Validation Actual')
plt.plot(range(0, len(pred_val)), pred_val, '-', label='Validation Predict')

plt.title('Prediction of House Prices')

plt.ylabel('Prices')
plt.legend()
```



[![img](https://1.bp.blogspot.com/-xnmCFFyAPuc/Xe3-qUKUytI/AAAAAAAAAEw/qHUz6BcDAS0UedJ9mw9zT2sKFCRR7JAYwCK4BGAYYCw/s640/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%2B2019-12-09%2B%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE%2B4.58.25.png)](https://1.bp.blogspot.com/-xnmCFFyAPuc/Xe3-qUKUytI/AAAAAAAAAEw/qHUz6BcDAS0UedJ9mw9zT2sKFCRR7JAYwCK4BGAYYCw/s1600/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%2B2019-12-09%2B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2B4.58.25.png)



로지스틱 회귀를 통해 예측한 값과 실제 값을 비교한 그래프이며, 파란색은 실제 값, 노란색은 예측 값을 의미합니다. Y축은 주택 가격을 나타내고, X축은 데이터의 번호입니다. XGBoost의 그래프와 비교해보았을 때, 실제 값과 예측 값이 극명한 차이를 보이는 부분이 확연히 많은 것을 볼 수 있습니다.

