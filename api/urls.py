from django.urls import path
from api.views.example_view import ExampleView
from api.views.image_decode_view import ImgDecodeView
app_name = 'api'

urlpatterns = [
    path('example', ExampleView.as_view(), name='example'),
    path('face', ImgDecodeView.as_view(), name='face'),
    
]

