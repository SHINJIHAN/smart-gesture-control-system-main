# 메시지 전송 API

AI-Lab 네트워크 내에서 다음과 같이 엔드포인트를 호출하세요.

```python
import requests

url = "http://192.168.0.8:8000/api/message/send"

payload = { "message": "[여기에 메시지 내용 입력]" }
headers = {"content-type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
```

**경고: for, while문 내에서 메시지 전송 시 주의하세요. 단기간에 빠른 메시지 전송 시 서버가 다운될 수 있습니다.**

