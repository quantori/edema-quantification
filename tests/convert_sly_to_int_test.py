import os

from src.data.convert_sly_to_int import create_save_dirs
from src.data.utils_sly import get_box_sizes, get_class_name, get_object_box, get_tag_value

dataset_dir_test = './../data/sly/DS1'
dataset_ann_dir_test = os.path.join(dataset_dir_test, 'ann')
dataset_img_dir_test = os.path.join(dataset_dir_test, 'img')

save_dir_test = './../data/intermediate_test'
save_ann_dir_test = os.path.join(save_dir_test, 'ann')
save_img_dir_test = os.path.join(save_dir_test, 'img')

ann_test_ok = {
    'description': '',
    'tags': [
        {
            'id': 133150989,
            'tagId': 30540785,
            'name': 'Stage',
            'value': 'Alveolar edema',
            'labelerLogin': 'RenataS',
            'createdAt': '2022-10-17T13:00:24.575Z',
            'updatedAt': '2022-10-17T13:00:24.575Z',
        },
    ],
    'size': {'height': 2000, 'width': 3328},
    'objects': [
        {
            'id': 1066227065,
            'classId': 10421845,
            'description': '',
            'geometryType': 'rectangle',
            'labelerLogin': 'RenataS',
            'createdAt': '2022-10-17T12:54:07.876Z',
            'updatedAt': '2022-10-23T16:44:41.725Z',
            'tags': [],
            'classTitle': 'Heart',
            'points': {'exterior': [[485, 915], [1260, 985]], 'interior': []},
        },
    ],
}

ann_test_no_tags = {
    'description': '',
    'tags': [],
    'size': {'height': 2000, 'width': 3328},
    'objects': [
        {
            'id': 1066226974,
            'classId': 10419435,
            'description': '',
            'geometryType': 'bitmap',
            'labelerLogin': 'RenataS',
            'createdAt': '2022-10-17T12:53:12.774Z',
            'updatedAt': '2022-10-23T16:44:41.725Z',
            'tags': [],
            'classTitle': 'Bronchus',
            'bitmap': {
                'data': 'eJzrDPBz5+WS4mJgYOD19HAJAtJCIMzBBiTDzvWFAynG4iB3J4Z152ReAjkZni6OIRb+Z6c4cjEocLDc/m9/y88to/tq2YnFLww6usJaZt6MYWJYPS21SG+Ko6CA7iJJ00mOog0zVZUlS7g80uZNcNQ64PqB98R3TzOpBbcTL5+1mxtaqSRa8PvX/MYbZ2acvKH87XrZ/wnfyxPbPZi/Cq17BbSTwdPVz2WdU0ITAASRP5U=',
                'origin': [479, 615],
            },
        },
    ],
}

object_test_ok = {
    'id': 1066226547,
    'classId': 10416423,
    'description': '',
    'geometryType': 'line',
    'labelerLogin': 'RenataS',
    'createdAt': '2022-10-17T12:47:37.167Z',
    'updatedAt': '2022-10-23T16:44:41.725Z',
    'tags': [
        {
            'id': 158526170,
            'tagId': 30540784,
            'name': 'RP',
            'value': '3',
            'labelerLogin': 'RenataS',
            'createdAt': '2022-10-17T12:49:03.635Z',
            'updatedAt': '2022-10-17T12:49:03.635Z',
        },
    ],
    'classTitle': 'Kerley',
    'points': {
        'exterior': [[237, 1018], [298, 1009], [262, 1016]],
        'interior': [],
    },
}

object_test_no_tags = {
    'id': 1066226974,
    'classId': 10419435,
    'description': '',
    'geometryType': 'bitmap',
    'labelerLogin': 'RenataS',
    'createdAt': '2022-10-17T12:53:12.774Z',
    'updatedAt': '2022-10-23T16:44:41.725Z',
    'tags': [],
    'classTitle': 'Bronchus',
    'bitmap': {
        'data': 'eJzrDPBz5+WS4mJgYOD19HAJAtJCIMzBBiTDzvWFAynG4iB3J4Z152ReAjkZni6OIRb+Z6c4cjEocLDc/m9/y88to/tq2YnFLww6usJaZt6MYWJYPS21SG+Ko6CA7iJJ00mOog0zVZUlS7g80uZNcNQ64PqB98R3TzOpBbcTL5+1mxtaqSRa8PvX/MYbZ2acvKH87XrZ/wnfyxPbPZi/Cq17BbSTwdPVz2WdU0ITAASRP5U=',
        'origin': [479, 615],
    },
}

object_test_polyline = {
    'id': 1066226547,
    'classId': 10416423,
    'description': '',
    'geometryType': 'line',
    'labelerLogin': 'RenataS',
    'createdAt': '2022-10-17T12:47:37.167Z',
    'updatedAt': '2022-10-23T16:44:41.725Z',
    'tags': [
        {
            'id': 158526170,
            'tagId': 30540784,
            'name': 'RP',
            'value': '3',
            'labelerLogin': 'RenataS',
            'createdAt': '2022-10-17T12:49:03.635Z',
            'updatedAt': '2022-10-17T12:49:03.635Z',
        },
    ],
    'classTitle': 'Kerley',
    'points': {
        'exterior': [[237, 1018], [298, 1009], [262, 1016]],
        'interior': [],
    },
}

object_test_polygon = {
    'id': 1066227212,
    'classId': 10416424,
    'description': '',
    'geometryType': 'polygon',
    'labelerLogin': 'RenataS',
    'createdAt': '2022-10-17T12:56:08.836Z',
    'updatedAt': '2022-10-23T16:44:41.725Z',
    'tags': [
        {
            'id': 158526260,
            'tagId': 30540784,
            'name': 'RP',
            'value': '3',
            'labelerLogin': 'RenataS',
            'createdAt': '2022-10-17T12:58:01.501Z',
            'updatedAt': '2022-10-17T12:58:01.501Z',
        },
    ],
    'classTitle': 'Bat',
    'points': {
        'exterior': [
            [1042, 533],
            [1065, 465],
            [1358, 997],
            [1362, 1030],
            [1271, 1028],
            [1255, 997],
            [1047, 642],
            [1034, 579],
        ],
        'interior': [],
    },
}

object_test_bitmap = {
    'id': 1066226974,
    'classId': 10419435,
    'description': '',
    'geometryType': 'bitmap',
    'labelerLogin': 'RenataS',
    'createdAt': '2022-10-17T12:53:12.774Z',
    'updatedAt': '2022-10-23T16:44:41.725Z',
    'tags': [],
    'classTitle': 'Bronchus',
    'bitmap': {
        'data': 'eJzrDPBz5+WS4mJgYOD19HAJAtJCIMzBBiTDzvWFAynG4iB3J4Z152ReAjkZni6OIRb+Z6c4cjEocLDc/m9/y88to/tq2YnFLww6usJaZt6MYWJYPS21SG+Ko6CA7iJJ00mOog0zVZUlS7g80uZNcNQ64PqB98R3TzOpBbcTL5+1mxtaqSRa8PvX/MYbZ2acvKH87XrZ/wnfyxPbPZi/Cq17BbSTwdPVz2WdU0ITAASRP5U=',
        'origin': [479, 615],
    },
}

object_test_rectangle = {
    'id': 1066227065,
    'classId': 10421845,
    'description': '',
    'geometryType': 'rectangle',
    'labelerLogin': 'RenataS',
    'createdAt': '2022-10-17T12:54:07.876Z',
    'updatedAt': '2022-10-23T16:44:41.725Z',
    'tags': [],
    'classTitle': 'Heart',
    'points': {'exterior': [[485, 915], [1260, 985]], 'interior': []},
}


def test_create_save_dirs():
    save_dir_img, save_dir_ann = create_save_dirs(save_dir=save_dir_test)
    assert save_dir_img == save_img_dir_test
    assert save_dir_ann == save_ann_dir_test


def test_get_class_name_ok():
    assert get_class_name(ann_test_ok) == 'Alveolar edema'


def test_get_class_name_wrong():
    assert get_class_name(ann_test_no_tags) == ''


def test_get_tag_value(caplog):
    assert get_tag_value(object_test_ok, 'RP') == '3'
    get_tag_value(object_test_no_tags, 'RP')
    assert 'No RP value in' in caplog.text


def test_get_object_box():
    assert get_object_box(object_test_polyline) == {'x1': 237, 'y1': 1009, 'x2': 298, 'y2': 1018}
    assert get_object_box(object_test_polygon) == {'x1': 1034, 'y1': 465, 'x2': 1362, 'y2': 1030}
    assert get_object_box(object_test_bitmap) == {'x1': 479, 'y1': 615, 'x2': 497, 'y2': 633}
    assert get_object_box(object_test_rectangle) == {'x1': 485, 'y1': 915, 'x2': 1260, 'y2': 985}


def test_get_box_sizes():
    assert get_box_sizes(0, 0, 0, 0) == {'xc': 0, 'yc': 0, 'Box width': 0, 'Box height': 0}
    assert get_box_sizes(0, 0, 1, 1) == {'xc': 0, 'yc': 0, 'Box width': 1, 'Box height': 1}
    assert get_box_sizes(1, 0, 9, 8) == {'xc': 4, 'yc': 4, 'Box width': 8, 'Box height': 8}
