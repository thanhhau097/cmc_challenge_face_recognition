from django.urls import path
from api.views.example_view import ExampleView
from api.views.image_decode_view import ImgDecodeView
from api.views.save_view import SaveView


app_name = 'api'

urlpatterns = [
    path('example', ExampleView.as_view(), name='example'),
    path('face', ImgDecodeView.as_view(), name='face'),
    path('save', SaveView.as_view(), name='save'),
]

