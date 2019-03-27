from django import forms
from api.forms.abstract_form import AbstractForm

class ImgDecodeForm(AbstractForm):
    base64_img = forms.CharField(required=True)