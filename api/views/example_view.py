from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView

from api.forms.example_form import ExampleForm
from api.helpers.response_format import json_format
from api.helpers.server.example_helper import processing


class ExampleView(APIView):
    parser_classes = (MultiPartParser,)
    success = 'Example Success'
    failure = 'Example Failed'

    def post(self, request):
        form = ExampleForm(request.POST,)
        if not form.is_valid():
            return JsonResponse(form.errors, status=422)
        number = form.cleaned_data.get('number')
        return self._format_response(number)
    
    def _format_response(self, number):
        result = processing(number)
        return json_format(code=200, message=self.success, data=result, errors=None)
  