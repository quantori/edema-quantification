import pytest

from tools.convert_sly_dataset import *

dataset_dir_test = './../../dataset/MIMIC-CXR-Edema-SLY/DS1'
dataset_ann_dir_test = os.path.join(dataset_dir_test, 'ann')
dataset_img_dir_test = os.path.join(dataset_dir_test, 'img')

save_dir_test = './../../dataset/MIMIC-CXR-Edema-Convert-TEST'
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
        }
    ],
    'size': {'height': 2000, 'width': 3328},
    'objects': [
        {
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
                }
            ],
            'classTitle': 'Kerley',
            'points': {
                'exterior': [[237, 1018], [298, 1009], [262, 1016]],
                'interior': [],
            },
        },
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
        {
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
                }
            ],
            'classTitle': 'Bat',
            'points': {
                'exterior': [
                    [1042, 533],
                    [1065, 465],
                    [1082, 411],
                    [1147, 443],
                    [1358, 997],
                    [1362, 1030],
                    [1333, 1034],
                    [1271, 1028],
                    [1255, 997],
                    [1065, 683],
                    [1047, 642],
                    [1034, 579],
                ],
                'interior': [],
            },
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
        }
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


def test_dataset_dir_exist():
    check_dataset_dirs(dataset_dir_test, dataset_ann_dir_test, dataset_img_dir_test)


def test_dataset_dir_not_exist():
    with pytest.raises(OSError):
        check_dataset_dirs('wrong', dataset_ann_dir_test, dataset_img_dir_test)


def test_dataset_ann_dir_not_exist():
    with pytest.raises(OSError):
        check_dataset_dirs(dataset_dir_test, 'wrong', dataset_img_dir_test)


def test_dataset_img_dir_not_exist():
    with pytest.raises(OSError):
        check_dataset_dirs(dataset_dir_test, dataset_ann_dir_test, 'wrong')


def test_crop_images():
    crop_images(dataset_img_dir_test, save_dir_test)

    original_img_list = os.listdir(dataset_img_dir_test)
    cropped_img_list = os.listdir(save_img_dir_test)

    assert len(original_img_list) == len(cropped_img_list)
    for original_img_name in original_img_list:
        subject_id, study_id, left_width, right_width, ext = original_img_name.replace(
            '.', '_'
        ).split('_')
        cropped_img_name = f'{subject_id}_{study_id}.{ext}'
        cropped_img_path = os.path.join(save_img_dir_test, cropped_img_name)
        assert os.path.isfile(cropped_img_path)

        cropped_img = Image.open(cropped_img_path)
        assert int(left_width) == cropped_img.width
        assert cropped_img.height == 2000


def test_get_edema_name_ok():
    assert get_edema_name(ann_test_ok) == 'Alveolar edema'


def test_get_edema_name_wrong():
    assert get_edema_name(ann_test_no_tags) == ''


def test_check_labeler_login(caplog):
    for bad_login in ('ViacheslavDanilov', 'mak_en', 'irina.ryndova'):
        check_labeler_login(bad_login)
        assert f'Wrong labeler login: {bad_login}' in caplog.text

    good_login = 'RenataS'
    check_labeler_login(good_login)
    assert f'Wrong labeler login: {good_login}' not in caplog.text


def test_get_object_rp(caplog):
    assert get_object_rp(object_test_ok) == '3'

    get_object_rp(object_test_no_tags)
    assert 'There is info about RP!' in caplog.text


def test_metadata():
    prepare_metadata_annotations(dataset_ann_dir_test, save_dir_test)
