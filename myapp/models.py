from django.db import models

class UserInfo(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    country_of_interest = models.CharField(max_length=100)
    interests = models.TextField()
    voice_type = models.CharField(max_length=50)

class Story(models.Model):
    user_info = models.ForeignKey(UserInfo, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField()  # JSON 대신 일반 텍스트로 저장
    voice_type = models.CharField(max_length=50)

class StoryImage(models.Model):
    story = models.ForeignKey(Story, related_name='images', on_delete=models.CASCADE)
    image_url = models.URLField()
    page_number = models.IntegerField()