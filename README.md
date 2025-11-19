# smart-gesture-control-system
**카메라 기반 손 제스처를 인식하여 장치를 제어하고 카카오톡 메시지를 전송하는 시스템**
> 본 프로젝트는 [hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe) 모델을 기반으로 합니다.
---

## [1] Python 3.10 기반 프로젝트 환경 구성
Smart Gesture Control System(SGCS) 프로젝트를 실행하기 위한 환경 구성 방법을 안내합니다.

### 1. Python 3.10 설치 확인
* 권장 버전: `python-3.10.8-amd64`
* 설치 확인 명령:
```bash
py --list
```
* 출력에 `-3.10-64`가 표시되어야 합니다.
---
### 2. 가상환경 생성
* 프로젝트 전용 가상환경 생성:
```bash
py -3.10 -m venv venv310
```
---
### 3. 가상환경 활성화
* Windows 기준:
```bash
venv310\Scripts\activate
```
* 활성화되면 프롬프트에 `(venv310)` 표시됨
---
### 4. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```
* 설치 후, `pip list`로 설치 여부 확인 가능
---
### 5. 프로젝트 실행
```bash
python app.py
```
* 실행 시 카메라 연결 및 제스처 인식 UI가 나타납니다.
---
## [2] Commit Message 규칙
| Tag         | 의미              | 사용 예시             |
| ----------- | --------------- | ----------------- |
| **[CHORE]** | 환경/설정 변경, 폴더 정리 | [CHORE]`.gitignore` 업데이트 |
| **[DOC]**   | README/문서/주석 수정 | [DOC] README 실행 방법 추가   |
| **[FIX]**   | 버그 수정           | [FIX] 제스처 인식 오류 수정      |
| **[FEAT]**  | 새로운 기능 추가       | [FEAT] 새로운 제스처 분류 추가     |
| **[DATA]**  | 데이터 관련          | [DATA] 제스처 학습용 데이터 추가  |
---