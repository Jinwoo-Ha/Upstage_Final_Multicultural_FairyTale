# views.py
from django.shortcuts import render, redirect
from .models import UserInfo, Story, StoryImage
from .story_generator import generate_story_with_images
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chatbot import ask_chatbot
from threading import Thread
import json

@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        response = ask_chatbot(user_input)
        return JsonResponse({'response': response})

def landing(request):
    return render(request, 'myapp/landing.html')

def storyinfo(request):
    if request.method == 'POST':
        user_info = UserInfo.objects.create(
            name=request.POST['name'],
            age=request.POST['age'],
            country_of_interest=request.POST['country'],
            interests=request.POST['interests'],
            voice_type=request.POST['voice_type']
        )
        return redirect('loading')
    return render(request, 'myapp/storyinfo.html')

def loading(request):
    user_info = UserInfo.objects.last()
    
    story = Story.objects.create(
        user_info=user_info,
        title=f"{user_info.name}의 {user_info.country_of_interest} 여행",
        content="",
        voice_type=user_info.voice_type
    )
    
    generate_story_async(story.id, user_info)
    
    return render(request, 'myapp/loading.html', {'story_id': story.id})

def check_story_status(request, story_id):
    story = Story.objects.get(id=story_id)
    status = 'completed' if story.content else 'in_progress'
    return JsonResponse({'status': status})

def story(request, story_id):
    story = Story.objects.get(id=story_id)
    images = story.images.order_by('page_number')
    pages = story.content.split('\n\n')
    content = [{'text': pages[i], 'image_url': images[i].image_url if i < len(images) else ''} 
               for i in range(len(pages))]
    content_json = json.dumps(content)
    return render(request, 'myapp/story.html', {'story': story, 'content': content_json})

def end(request):
    return render(request, 'myapp/end.html')

def generate_story_async(story_id, user_info):
    def task():
        story_content, images = generate_story_with_images(
            name=user_info.name,
            age=user_info.age,
            country=user_info.country_of_interest,
            interests=user_info.interests
        )
        story = Story.objects.get(id=story_id)
        story.content = story_content
        story.save()
        
        for i, image_url in enumerate(images):
            if image_url:  # image_url이 None이 아닌 경우에만 StoryImage 생성
                StoryImage.objects.create(story=story, image_url=image_url, page_number=i)
    
    Thread(target=task).start()

# Google cloud tts
from django.http import HttpResponse
from google.cloud import texttospeech
import json

def text_to_speech(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR", 
            name="ko-KR-Neural2-C"
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        return HttpResponse(response.audio_content, content_type='audio/mpeg')
    
    return HttpResponse(status=405)  # Method Not Allowed