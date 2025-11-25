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
* 활성화되면 프롬프트에 `(venv310)` 표시된다.
---
### 4. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```
* 설치 후, `pip list`로 설치 여부 확인 가능.
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
## [3] 협업 흐름
### 1️. 원격 브랜치 최신 정보 가져오기
```bash
git fetch
```
* 원격 저장소의 최신 커밋 정보를 로컬로 가져옵니다.
* VS코드에서는 **Git: Auto Fetch (all)** 를 켜면 주기적으로 자동으로 fetch된다.
---
### 2️. 로컬 브랜치 생성
```bash
git checkout -b <로컬브랜치> origin/<원격브랜치>
```
* 원격 브랜치를 기준으로 **로컬 브랜치 생성**
* 격 브랜치의 최신 상태를 기준으로 작업할 수 있다.
* 주의: `git checkout origin/<브랜치>`만 하면 detached HEAD 상태가 된다.
---
### 3️. 작업 후 커밋
```bash
git add .
git commit -m "작업 내용 설명"
```
* 변경사항을 스테이징하고 로컬에 커밋.
* 커밋은 로컬에서만 존재하며, 팀원에게는 아직 보이지 않는다.
---
### 4️. 원격 브랜치에 반영 (팀원이 볼 수 있도록)
```bash
git push origin <브랜치 이름>
```
* 로컬 커밋을 원격 브랜치에 올려 팀원들이 pull로 확인 가능.
* 다른 팀원의 변경사항은 **pull** 또는 **VS코드 Auto Fetch + pull**을 통해 로컬에 반영한다.
---
## [4] 구현 내용

### 1. 기능 구현
- **3번 키**: 왼손 OK 제스처 미구현 상태에서 기능 구현 완료 (jihan).
- **4번 키**: 각 손의 Call 제스처 구현 완료 (choi).
- **5번 키**: 두 손을 사용한 Olise 제스처 구현 완료 (ahn).

### 2. UI 개선
- 전체화면 모드 지원 가능한 창 형태로 수정 (choi).
---
