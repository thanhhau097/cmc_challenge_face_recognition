from django import forms
from api.forms.abstract_form import AbstractForm

class SaveForm(AbstractForm):
    base64_img = forms.CharField(required=True)
    image_name = forms.CharField(required=True)