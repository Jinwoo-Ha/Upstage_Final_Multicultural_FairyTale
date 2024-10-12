from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('storyinfo/', views.storyinfo, name='storyinfo'),
    path('loading/', views.loading, name='loading'),
    path('story/<int:story_id>/', views.story, name='story'),
    path('end/', views.end, name='end'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('check_story_status/<int:story_id>/', views.check_story_status, name='check_story_status'),
    #google cloud tts
    path('tts/', views.text_to_speech, name='text_to_speech'),
    path('languagestudy/', views.languagestudy, name='languagestudy'),  # 새로운 URL 패턴 추가

]