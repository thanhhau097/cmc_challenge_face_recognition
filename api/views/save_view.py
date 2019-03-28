from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView
from api.forms.image_decode_form import ImgDecodeForm
from api.helpers.image_decode_helper import decode_base64, get_bbox, get_result
from api.helpers.response_format import json_format

class SaveView(APIView):
    parser_classes = (JSONParser, MultiPartParser)
    #parser_classes = (MultiPartParser,)
    success = 'Success'
    failure = 'Failed'

    def post(self, request):
        form = ImgDecodeForm(request.data,)
        # form = ImgDecodeForm(request.POST, )
        if not form.is_valid():
            return JsonResponse(form.errors, status=422)
        base64_img = form.cleaned_data.get('base64_img')
        return self._format_response(base64_img)

    def _format_response(self, base64_img):
        img = decode_base64(base64_img)
        res = json_format(code=200, message=self.success, data=1, errors=None)
        return res
        # return json_format(code=500, message=self.failure, data={"result": self.failure}, errors=None)


