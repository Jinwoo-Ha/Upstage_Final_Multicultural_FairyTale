import requests 
import json
import wave
import pyaudio
import io
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://typecast.ai/api/speak"
text_to_convert = "안녕하세요, 이것은 VSCode에서 실행한 Typecast API 테스트입니다. 파일을 저장하지 않고 직접 재생합니다."

payload = json.dumps({
  "actor_id": "6080369d3211aa112ab131db",
  "text": text_to_convert,
  "lang": "auto",
  "tempo": 1,
  "volume": 100,
  "pitch": 0,
  "xapi_hd": True,
  "max_seconds": 60,
  "model_version": "latest",
  "xapi_audio_format": "wav"
})

headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Bearer {os.getenv("TYPECAST_API_KEY")}'
}

try:
    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
    print("응답 상태 코드:", response.status_code)
    
    if response.status_code == 200:
        print("오디오 데이터를 받았습니다. 재생을 시작합니다...")
        
        # 응답 내용을 BytesIO 객체로 변환
        audio_data = io.BytesIO(response.content)
        
        # WAV 파일 열기
        with wave.open(audio_data, 'rb') as wf:
            # PyAudio 객체 초기화
            p = pyaudio.PyAudio()
            
            # 스트림 열기
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            
            # 청크 단위로 데이터 읽고 재생
            chunk = 1024
            data = wf.readframes(chunk)
            while data:
                stream.write(data)
                data = wf.readframes(chunk)
            
            # 스트림 정리
            stream.stop_stream()
            stream.close()
            
            # PyAudio 객체 정리
            p.terminate()
        
        print("오디오 재생이 완료되었습니다.")
    
except requests.exceptions.RequestException as e:
    print("API 요청 중 오류가 발생했습니다:", e)
except Exception as e:
    print("오디오 재생 중 오류가 발생했습니다:", e)