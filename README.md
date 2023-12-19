### Project Title

만화책 번역 프로젝트(NLP/Vision 결)  

### Overview

- 기간  |  2021. 06 ~ 2021. 06
- 담당 파트 |  개인프로젝트
- 플랫폼 |  Python, Tensorflow, Colab notebook

### Background 

1. 플랫폼의 발달에 따라 만화, 영화, 동영상 등 다양한 콘텐츠가 글로벌하게 공유되고 있음
2. 말풍선, 자막 등의 번역 작업은 콘텐츠 교류의 중요 요소이나, 작업 특성 상 시간과 비용, 노동력이 요구되며, 번역 작업이 이루어지지 않은 콘텐츠도 다수 존재
3. 기계번역에서 훌륭한 성과를 내고 있는 Transformer 모델을 토대로 기계번역 성능과 기대효과를 도출하고자 함
4. 만화 Text 인식 및 번역 모델 구현을 통해 콘텐츠 번역 모델의 상용화 가능성을 살펴보고자 함

### Goal 

1. Transformer 기계번역 모델 구현
2. 식질머신, Google OCR을 활용한 Text 인식 모델 구현 및 최종 번역
   
### Dataset

- KAIST 제공 한영 병렬 데이터 60000문장

### Theories

1.  Transfromer 모델은 Base 모델(LSTM)보다 BLEU Score가 10% 이상 높을 것이다.
2. Subword Tokenizer를 사용한 모델 성능이 그렇지 않은 모델보다 BLEU Score가 10% 이상 높을 것이다. 

### LSTM(Long Short-Term Memory)

1. 바닐라 RNN의 장기 의존성 문제(the problem of Long-Term Dependencies)를 해결하기 위한 모델
2. 은닉층 메모리셀에 3개의 GATE 추가
   - forget gate : 과거 정보의 유지를 담당
   - input gate :  입력된 정보의 활용을 담당
   - output gate : 두 정보를 계산하여 나온 출력 정보를 담당
3. Cell-state 추가
   - 활성화 함수를 거치지 않기에 정보손실이 없음
   - 최근(short) 이벤트에 비중을 결정할 수 있으면서 동시에 오래된(long) 정보를 완전히 잃지 않을 수 있음
4. 교사강요(Teacher forcing) 적용
   - 모든 시점에 대해서 이전 시점의 예측값 대신 실제값을 입력으로 주는 방법

<p align="center">
  <img src="https://github.com/mugan1/comic_translation/assets/71809159/c41f3790-9f78-4d8d-9251-b96059a9d14b" alt="text" width="number" />
</p>

### Transformer

1. RNN 모델은 단어가 순서대로 들어오기 때문에 입력 시퀀스의 정보가 소실되는 구조적 한계점이 있음
2. 모든 토큰을 동시에 입력받아 병렬 연산하는 방식
3. N개의 인코더와 디코더로 구성
4. 디코더에서 출력 단어를 예측하는 매 시점마다, 인코더에서의 전체 입력 문장을 다시 한 번 참고
5. 해당 시점에서 예측해야 할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중(attention)함
6. Attention 연산 과정
   - Query에 대해 Key와의 유사도를 가중치로 하여 Key와 mapping된 Value에 반영함. 이후 이들의 가중합을 리턴
   - Self-Attention은 입력 문장의 모든 단어 벡터들끼리 Attention을 적용(인코더)
   - Masked Decoder Attention은 디코더가 출력을 할 때 다음 정보를 미리 얻지 못하게 하기 위해 Masking을 함(디코더)
   - Encoder-Decoder Attention은 Query는 디코더에, Key와 Value는 인코더가 출처로서 이들의 가중합을 구함(디코더)
     
<p align="center">
   <img src="https://github.com/mugan1/comic_translation/assets/71809159/b2eb7f16-6bcc-4035-b2a5-39212f8672d0" alt="text" width="number" />
</p> 

### BLEU Score(Bilingual Evaluation Understudy Score)

1. BLEU는 기계 번역 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하여 번역에 대한 성능을 측정하는 방법
2. 1-gram부터 가중치를 균일하게 적용한 1-4-grams까지 BLEU Score 계산 후 비교

### Tokenizer

1. 한국어 형태소 분석기인 Konlpy의 Komoran 및 Mecab 사용 후 성능 비교
2. Mecab과 Subword Tokenizer인 Huggingface Tokenzier로 토큰화 한 후 성능 비교
3. Subword Tokenizer 
   - OOV(Out-Of-Vocabulary) 문제를 해결하기 위해 하나의 단어를 더 작은 단위의 의미있는 여러 서브워드로 분할하는 방식
   - Ex) ['나', '는', '오늘', '아침밥', '을', '먹', '었', '다'] → ['나', '##는', '오늘', '아침', '##밥', '##을', '먹', '##었다', '.']
  
### Translation 결과 및 분석  

1. Train Data

   1)<br>
      - 원문 :이렇게 건들건들 돌아다니지 말고 좀 얌전히 일을 좀 해라.
      - 번역문 : oh stop all this gallivanting aboutand settle down to something!
      - LSTM :newspaper problem i 'm afraid so long it 's not be so small
      - Transformer(Mecab) : oh stop all this is !
      - **Transformer(Subword) : oh stop all this gallivanting aboutand settle down to something!**
   

   2)<br>
      - 원문 : 내가 인내하려는 것에 대하여 훈계하지 마라.
      - 번역문 : don't preach me a lesson about patience
      - LSTM :excuse for his friends
      - Transformer(Mecab) : don't preach me a lesson about patience
      - **Transformer(Subword) : don't preach me a lesson about patience**

2. Test Data

   - 원문 :그들은 사람들이 그들의 요구에 대하여 핑계대어 거절하고 회피하는 것을 허락치 않았다.
   - 번역문 : they refused to have their demands put off.
   - LSTM : they had to send their own some letters to the enemy 's own ow
   - Transformer(Mecab) : they did not allow any excuse to permit them on their demands to further language.
   - **Transformer(Subword) : they refused to excuse their demands for their demands**

<p align="center">
   <img src="https://github.com/mugan1/comic_translation/assets/71809159/d892af29-79c8-4c56-a371-617bce07e33d" alt="text" width="number" />
</p> 

<p align="center">
   <img src="https://github.com/mugan1/comic_translation/assets/71809159/dcb1ca00-c772-4740-a627-eea14333ecc4" alt="text" width="number" />
</p> 

### Analysis

1. Train Data에 한해서 Subword Tokenizer를 이용한 Transformer 모델이 성능이 가장 높았으며, 기준 모델인 LSTM보다 최소 4배 이상의 높은 BLEU Score를 달성
2. Test Data에서는 과적합 현상을 보이며 모든 모델이 제대로 된 성능을 발휘하지 못했는데, 언어의 복잡성을 고려했을 때 데이터셋과 학습량이 절대적으로 부족했기 때문이라고 판단함

### 만화 Text 감지 및 데이터 변환 프로세스

<p align="center">
   <img src="https://github.com/mugan1/comic_translation/assets/71809159/f9acee3b-3960-42a3-86b6-c45a0915321e" alt="text" width="number" />
</p> 

1. OpenCV를 이용한 Text Detection 수행 
2. Google OCR을 활용하여 Text 이미지를 text 데이터로 변환
3.  번역된 결과를 Image draw를 통해 텍스트 이미지 삽입

<p align="center">
   <img src="https://github.com/mugan1/comic_translation/assets/71809159/02dc9a2a-39e3-49ff-9f46-fcef1d2d911a" alt="text" width="number" />
</p> 

### Result



- Framework |  FLASK
- DB |  SQLITE3, POSTGRE(Elephant SQL)
- Web Hosting | Koyeb

### Layout

1. 사용자 입력화면 : 사용자 입력데이터와 모델 예측 결과는 Input Table DB에 저장됨
   
<p align="center">
  <img src="https://github.com/mugan1/Used_Car_Prediction/assets/71809159/17040b86-3608-411e-bf5c-b4cac2986ccb" alt="text" width="number" /><br>
</p>   

2. 대시보드 : 모델을 통해 예측한 가격과 관련 분석 정보를 확인할 수 있음

<p align="center">
  <img src="https://github.com/mugan1/Used_Car_Prediction/assets/71809159/c9b24b46-8198-4e48-b6be-b80d67592608" alt="text" width="number" /><br>
</p>  
<p align="center">
  <img src="https://github.com/mugan1/Used_Car_Prediction/assets/71809159/fb88abd7-82af-4759-8a45-a41158870b4f" alt="text" width="number" /><br>
</p>   

### Conclusion

- 직접 웹스크래핑으로 수집한 데이터를 분석하고 모델링하여 웹 애플리케이션으로 배포하는 프로젝트를 수행함으로써, ML 엔지니어링에 대한 이해를 한층 더 키울 수 있었음
- 데이터량의 부족과 모델 하이퍼파라미터 최적화 문제, 변수 선택의 문제로 좋은 성능의 모델을 만들지 못했는데, 차후 수정을 통해 R2 Score 80% 이상의 모델을 구현하여 교체할 예정임

