# script for creating a data validation report with `deepchecks`

from deepchecks.vision.simple_classification_data import load_dataset
from deepchecks.vision.suites import train_test_validation
from config import PROCESSED_IMAGES_DIR

train_ds = load_dataset(PROCESSED_IMAGES_DIR, train=True, object_type="VisionData", image_extension="jpg")
test_ds = load_dataset(PROCESSED_IMAGES_DIR, train=False, object_type="VisionData", image_extension="jpg")

suite = train_test_validation()
result = suite.run(train_ds, test_ds)

result.save_as_html("data_validation.html")