"""
Supporting code for running SyntaxGym CLI tests.
"""

from functools import lru_cache, wraps
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile

import pytest

import docker
import docker.tls

# Silence a few loud modules
logging.getLogger("docker").setLevel(logging.ERROR)
import urllib3
logging.getLogger("urllib3").setLevel(logging.ERROR)

from syntaxgym import utils


LM_ZOO_IMAGES_TO_BUILD = {
    "basic": "lmzoo-basic",
    "basic_eos_sos": "lmzoo-basic-eos-sos",
    "basic_eos_sos_same": "lmzoo-basic-eos-sos-same",
    "basic_uncased": "lmzoo-basic-uncased",
    "basic_nopunct": "lmzoo-basic-nopunct",
    "bert_tokenization": "lmzoo-bert-tokenization",
    "gpt_tokenization": "lmzoo-gpt-tokenization",
    "moses_tokenization": "lmzoo-moses-tokenization",
}
LM_ZOO_IMAGE_TO_DIRECTORY = {image: directory
                             for directory, image in LM_ZOO_IMAGES_TO_BUILD.items()}

# Images to test -- harness images and, optionally, real images
LM_ZOO_IMAGES = []
LM_ZOO_IMAGES.extend((image, "latest") for image in LM_ZOO_IMAGES_TO_BUILD.values())

BUILT_IMAGES = []


@pytest.fixture(scope="module")
def client():
    return utils._get_docker_client().api


@pytest.fixture(scope="module")
def built_image(client, request):
    ref_fields = request.param.rsplit(":", 1) if isinstance(request.param, str) else request.param
    print(ref_fields)
    if len(ref_fields) == 1:
        image, tag = ref_fields[0], "latest"
    else:
        image, tag = ref_fields

    build_image(client, image, tag)
    return ":".join((image, tag))


def build_image(client, image, tag="latest"):
    try:
        image_dir = LM_ZOO_IMAGE_TO_DIRECTORY[image]
    except KeyError:
        print("Image %s not found in dummy images directory. Skipping." % (image,), file=sys.stderr)
        return

    image_dir = Path(__file__).parent / "dummy_images" / image_dir

    out = client.build(path=str(image_dir), rm=True, tag=f"{image}:{tag}")

    BUILT_IMAGES.append(f"{image}:{tag}")
    ret = list(out)

    return ret


def with_images(*images):
    def decorator_with_images(fn):
        ids = [":".join(image_ref) if not isinstance(image_ref, str) else image_ref
               for image_ref in images]

        @pytest.mark.parametrize("built_image", images, ids=ids,
                                 indirect=["built_image"])
        @wraps(fn)
        def my_testfn(*args, **kwargs):
            return fn(*args, **kwargs)

        return my_testfn
    return decorator_with_images


_dummy_suite_json = {
    "meta": {
        "name": "subordination_orc-orc",
        "metric": "sum",
        "comment": "",
        "reference": "",
        "author": "",
    },
    "region_meta": {
        "1": "Subordinate clause 1",
        "2": "subj_modifier",
        "3": "Subordinate clause 2",
        "4": "obj_modifier",
        "5": "Main clause"
    },
    "predictions": [
        {"type": "formula",
            "formula": "((5;%sub_no-matrix%) > (5;%no-sub_no-matrix%) ) & ((5;%sub_matrix%) < (5;%no-sub_matrix%) )"}
    ],
    "items": [
        {
            "item_number": 1,
            "conditions": [
                {
                    "condition_name": "sub_no-matrix",
                    "regions": [
                        {
                            "region_number": 1,
                            "content": "After the man",
                            "metric_value": {
                                "sum": 18.504567167144707
                            },
                            "oovs": ["After", "man"]
                        },
                        {
                            "region_number": 2,
                            "content": "who a friend had helped",
                            "metric_value": {
                                "sum": 13.939815152698332
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 3,
                            "content": "shot the bird",
                            "metric_value": {
                                "sum": 11.948716490393831
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 4,
                            "content": "that he had been tracking secretly",
                            "metric_value": {
                                "sum": 11.217957010846014
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 5,
                            "content": ".",
                            "metric_value": {
                                "sum": 12.49306574242164
                            },
                            "oovs": []
                        }
                    ]
                },
                {
                    "condition_name": "no-sub_no-matrix",
                    "regions": [
                        {
                            "region_number": 1,
                            "content": "The man",
                            "metric_value": {
                                "sum": 2.268286909168587
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 2,
                            "content": "who a friend had helped",
                            "metric_value": {
                                "sum": 1.4433385639057028
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 3,
                            "content": "shot the bird",
                            "metric_value": {
                                "sum": 9.693795300311402
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 4,
                            "content": "that he had been tracking secretly",
                            "metric_value": {
                                "sum": 7.773922435893982
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 5,
                            "content": ".",
                            "metric_value": {
                                "sum": 18.37796694276843
                            },
                            "oovs": []
                        }
                    ]
                },
                {
                    "condition_name": "sub_matrix",
                    "regions": [
                        {
                            "region_number": 1,
                            "content": "After the man",
                            "metric_value": {
                                "sum": 10.356803538573178
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 2,
                            "content": "who a friend had helped",
                            "metric_value": {
                                "sum": 1.4248155193516632
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 3,
                            "content": "shot the bird",
                            "metric_value": {
                                "sum": 3.409407627322641
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 4,
                            "content": "that he had been tracking secretly",
                            "metric_value": {
                                "sum": 9.729441419245582
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 5,
                            "content": ", he loaded his gun .",
                            "metric_value": {
                                "sum": 11.725874091041778
                            },
                            "oovs": []
                        }
                    ]
                },
                {
                    "condition_name": "no-sub_matrix",
                    "regions": [
                        {
                            "region_number": 1,
                            "content": "The man",
                            "metric_value": {
                                "sum": 16.30488999162344
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 2,
                            "content": "who a friend had helped",
                            "metric_value": {
                                "sum": 4.9608373385007845
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 3,
                            "content": "shot the bird",
                            "metric_value": {
                                "sum": 5.193990201733378
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 4,
                            "content": "that he had been tracking secretly",
                            "metric_value": {
                                "sum": 5.375807689210841
                            },
                            "oovs": []
                        },
                        {
                            "region_number": 5,
                            "content": ", he loaded his gun .",
                            "metric_value": {
                                "sum": 19.63184731373726
                            },
                            "oovs": []
                        }
                    ]
                }
            ],
        },
    ],
}

@pytest.fixture(scope="session")
def dummy_suite_json():
    return _dummy_suite_json


_dummy_suite_csv = """item_number,condition_name,region_number,content,metric_value,oovs
1,sub_no-matrix,1,After the man,18.504567167144707,"After,man"
1,sub_no-matrix,2,who a friend had helped,13.939815152698332,
1,sub_no-matrix,3,shot the bird,11.948716490393831,
1,sub_no-matrix,4,that he had been tracking secretly,11.217957010846014,
1,sub_no-matrix,5,.,12.49306574242164,
1,no-sub_no-matrix,1,The man,2.268286909168587,
1,no-sub_no-matrix,2,who a friend had helped,1.4433385639057028,
1,no-sub_no-matrix,3,shot the bird,9.693795300311402,
1,no-sub_no-matrix,4,that he had been tracking secretly,7.773922435893982,
1,no-sub_no-matrix,5,.,18.37796694276843,
1,sub_matrix,1,After the man,10.356803538573178,
1,sub_matrix,2,who a friend had helped,1.4248155193516632,
1,sub_matrix,3,shot the bird,3.409407627322641,
1,sub_matrix,4,that he had been tracking secretly,9.729441419245582,
1,sub_matrix,5,", he loaded his gun .",11.725874091041778,
1,no-sub_matrix,1,The man,16.30488999162344,
1,no-sub_matrix,2,who a friend had helped,4.9608373385007845,
1,no-sub_matrix,3,shot the bird,5.193990201733378,
1,no-sub_matrix,4,that he had been tracking secretly,5.375807689210841,
1,no-sub_matrix,5,", he loaded his gun .",19.63184731373726,"""

@pytest.fixture(scope="session")
def dummy_suite_csv():
    return _dummy_suite_csv
