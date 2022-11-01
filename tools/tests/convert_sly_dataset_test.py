import pytest

from tools.convert_sly_dataset import *

dataset_dir_test = "./../../dataset/MIMIC-CXR-Edema-SLY/DS1"
dataset_ann_dir_test = os.path.join(dataset_dir_test, "ann")
dataset_img_dir_test = os.path.join(dataset_dir_test, "img")

save_dir_test = "./../../dataset/MIMIC-CXR-Edema-Convert-TEST"
save_ann_dir_test = os.path.join(save_dir_test, "ann")
save_img_dir_test = os.path.join(save_dir_test, "img")


def test_dataset_dir_exist():
    check_dataset_dirs(dataset_dir_test, dataset_ann_dir_test, dataset_img_dir_test)


def test_dataset_dir_not_exist():
    with pytest.raises(OSError):
        check_dataset_dirs("wrong", dataset_ann_dir_test, dataset_img_dir_test)


def test_dataset_ann_dir_not_exist():
    with pytest.raises(OSError):
        check_dataset_dirs(dataset_dir_test, "wrong", dataset_img_dir_test)


def test_dataset_img_dir_not_exist():
    with pytest.raises(OSError):
        check_dataset_dirs(dataset_dir_test, dataset_ann_dir_test, "wrong")


def test_crop_images():
    crop_images(dataset_img_dir_test, save_dir_test)

    original_img_list = os.listdir(dataset_img_dir_test)
    cropped_img_list = os.listdir(save_img_dir_test)

    assert len(original_img_list) == len(cropped_img_list)
    for original_img_name in original_img_list:
        subject_id, study_id, left_width, right_width, ext = original_img_name.replace(
            ".", "_"
        ).split("_")
        cropped_img_name = f"{subject_id}_{study_id}.{ext}"
        cropped_img_path = os.path.join(save_img_dir_test, cropped_img_name)
        assert os.path.isfile(cropped_img_path)

        cropped_img = Image.open(cropped_img_path)
        assert int(left_width) == cropped_img.width
        assert cropped_img.height == 2000


def test_metadata():
    prepare_metadata_annotations(dataset_ann_dir_test, save_dir_test)
