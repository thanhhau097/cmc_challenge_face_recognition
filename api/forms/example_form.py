from django import forms
from api.forms.abstract_form import AbstractForm

class ExampleForm(AbstractForm):
    number = forms.IntegerField(required=True, initial=0)