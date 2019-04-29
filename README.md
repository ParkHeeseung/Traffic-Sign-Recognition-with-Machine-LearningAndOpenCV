# Traffic-Sign-Recognition-with-OpenCV
효과적인 전처리를 통한 표지판 후보군을 뽑아낸 후 Feature Vector로 학습된 분류기에 넣어 표지판을 구별한다.

![d](https://user-images.githubusercontent.com/31951367/56875289-cda1d400-6a7a-11e9-90ee-2ec500e8160c.png)

# Image preprocessing
1. HSV변환 및 이진화
2. 후보군 검출(Contours & Standard)
3. 후보군 ROI 설정

분류기에 들어갈 후보군들
![candidates_screenshot_29 04 2019](https://user-images.githubusercontent.com/31951367/56879013-c76b2200-6a91-11e9-9f55-853595b28a3e.png)


# Classifier
KNN vs SVM

KNN

k-NN 알고리즘은 지도 학습(Supervised Learning)의 한 종류로 레이블이 있는 데이터를 사용하여 분류 작업을 하는 알고리즘이다. 알고리즘의 이름에서 볼 수 있듯이 데이터로부터 거리가 가까운 k개의 다른 데이터의 레이블을 참조하여 분류하는 알고리즘이다.

SVM

SVM 알고리즘은 데이터셋의 인스턴스가 다차원 공간의 점이라고 생각할 때, 서로 다른 범주에 속한 인스턴스 간의 거리를 최대한 크게 만드는 인스턴스들을 선택하는 방법으로 초평면을 얻는 지도 학습 기법으로 새 인스턴스는 놓여진 초평면의 촉을 기반으로 특정범주로 분류되는 알고리즘이다.

# Feature
Sift vs Surf vs Hog

Sift

Sift는 영상에서 코너점 등 식별이 용이한 특징점들을 선택한 후에 각 특징점을 중심으로 한 로컬패치에 대해 특징 벡터를 추출한 것으로 각 로컬패치의 속한 픽셀들의 gradient방향과 크기에 대한 히스토그램을 구한 후 bin값들을 일렬로 쭉 연결한 128차원 벡터이다.
스케일, 회전, 밝기변화에 대한 불변 특징을 위해 어느정도 구분력을 희생한다.

Surf

Surf는 Scale space 상에서 Hessian 행렬의 행렬식(determinant)이 극대인 점들을 특징점으로 검출한다.

Hog

Hog는 대상 영역을 일정 크기의 셀로 분할하고, 각 셀마다 edge 픽셀들의 방향에 대한 히스토그램을 구한 후 bin값들을 일렬로 연결한 벡터이다.
내부 패턴이 복잡하지 않고 고유의 독특한 윤곽선 정보를 갖는 물체를 식별하는데 적합한 feature이다.

# The progress of the work
1. KNN + Surf (80%)...

![선택 영역_001](https://user-images.githubusercontent.com/31951367/55569679-f31e1500-573c-11e9-9789-1c6bc55286e1.png)

2. KNN + Hog (95%)...
![선택 영역_020](https://user-images.githubusercontent.com/31951367/56080615-3ca5e880-5e3e-11e9-9f3c-d9d01e16095e.png)





