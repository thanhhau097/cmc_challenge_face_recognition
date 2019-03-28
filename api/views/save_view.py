from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView
from api.forms.save_form import SaveForm
from api.helpers.image_decode_helper import decode_base64, get_bbox, get_result
from api.helpers.save_image import get_and_save_embedding
from api.helpers.response_format import json_format

class SaveView(APIView):
    parser_classes = (JSONParser, MultiPartParser)
    #parser_classes = (MultiPartParser,)
    success = 'Success'
    failure = 'Failed'

    def post(self, request):
        # TODO
        form = SaveForm(request.data,)
        # form = ImgDecodeForm(request.POST, )
        if not form.is_valid():
            return JsonResponse(form.errors, status=422)
        base64_img = form.cleaned_data.get('base64_img')
        img_name = form.cleaned_data.get('image_name')

        return self._format_response(base64_img, img_name)

    def _format_response(self, base64_img, img_name):
        img = decode_base64(base64_img)
        get_and_save_embedding(image_name=img_name, image=img)
        res = json_format(code=200, message=self.success, data=1, errors=None)
        return res
        # return json_format(code=500, message=self.failure, data={"result": self.failure}, errors=None)


