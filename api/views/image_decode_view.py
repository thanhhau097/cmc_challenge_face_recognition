from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView
from api.forms.image_decode_form import ImgDecodeForm
from api.helpers.image_decode_helper import decode_base64, get_bbox, get_result
from api.helpers.response_format import json_format

class ImgDecodeView(APIView):
    parser_classes = (JSONParser,)
    #parser_classes = (MultiPartParser,)
    success = 'Decode Success'
    failure = 'Decode Failed'

    def post(self, request):
        form = ImgDecodeForm(request.data,)
        # form = ImgDecodeForm(request.POST, )
        if not form.is_valid():
            return JsonResponse(form.errors, status=422)
        base64_img = form.cleaned_data.get('base64_img')
        return self._format_response(base64_img)

    def _format_response(self, base64_img):
        img = decode_base64(base64_img)
        x1, y1, x2, y2 = get_bbox(img)
        list_res = get_result()
        bbox = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
        list_res = dict(list_res)
        new_dict = {}
        same_persons = []

        for key in list_res.keys():
            values = list_res[key]
            # for value in dict_rest[key]:
            bbox_detail = {'x1': int(values[1][0]), 'y1': int(values[1][1]), 'x2': int(values[1][2]), 'y2': int(values[1][3])}
            distance = float(values[0][0][0])

            same_persons.append({'path': key[:-1] + '.jpg', 'distance': distance, 'bbox_detail': bbox_detail})
            # new_dict[] = {'distance': distance, 'bbox_detail': bbox_detail}
            # list_res[key] = {'distance': distance, 'bbox_detail': bbox_detail}
        result = {"bbox": bbox, "same_person": same_persons}

        return json_format(code=200, message=self.success, data=result, errors=None)