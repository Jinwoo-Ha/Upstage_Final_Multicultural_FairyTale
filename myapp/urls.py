from django.urls import path
from . import views
from .views import (
    tts_view, chatbot_view, landing, storyinfo, loading, story, end, languagestudy, check_story_status,
)
from .auto_storyfilling import auto_fill


urlpatterns = [
    path('', views.landing, name='landing'),
    path('storyinfo/', views.storyinfo, name='storyinfo'),
    path('loading/', views.loading, name='loading'),
    path('story/<int:story_id>/', views.story, name='story'),
    path('end/', views.end, name='end'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('check_story_status/<int:story_id>/', views.check_story_status, name='check_story_status'),
    path('tts/', tts_view, name='tts'),
    path('languagestudy/', views.languagestudy, name='languagestudy'),  # 새로운 URL 패턴 추가
    path('auto_fill/', auto_fill, name='auto_fill'),

]