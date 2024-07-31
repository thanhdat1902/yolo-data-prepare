import os
import fiftyone as fo
import fiftyone.utils.data as foud

class YOLOv8PoseDatasetImporter(foud.LabeledImageDatasetImporter):
    def __init__(self, dataset_dir, label_dir, shuffle=False, seed=None, max_samples=None):
        super().__init__(dataset_dir=dataset_dir, shuffle=shuffle, seed=seed, max_samples=max_samples)
        self.label_dir = label_dir
        self._labels = None
        self._iter_labels = None

    def __iter__(self):
        self._iter_labels = iter(self._labels)
        return self

    def __next__(self):
        if self._iter_labels is None:
            raise ValueError("No labels loaded. Did you forget to call `setup()`?")

        try:
            label_file = next(self._iter_labels)
        except StopIteration:
            raise StopIteration

        label_path = os.path.join(self.label_dir, label_file)
        with open(label_path, "r") as f:
            data = f.readline().strip()
        
        class_id, x_center, y_center, bbox_width, bbox_height, *keypoints_data = map(float, data.split())

        image_file = label_file.replace(".txt", ".PNG")
        image_path = os.path.join(self.dataset_dir, image_file)

        keypoints = []
        for i in range(0, len(keypoints_data), 3):
            kp_x = keypoints_data[i]
            kp_y = keypoints_data[i + 1]
            visibility = int(keypoints_data[i + 2])

            keypoints.append(fo.Keypoint(
                points=[(kp_x, kp_y)],
                visibility=visibility
            ))

        return image_path, None, fo.Keypoints(keypoints=keypoints)

    def __len__(self):
        return len(self._labels)

    @property
    def has_image_metadata(self):
        return False
    @property
    def has_dataset_info(self):
        """Whether this importer provides additional dataset information."""
        return False
    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this
        importer.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            importer is guaranteed to return labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the importer can produce a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the importer will return label dictionaries with keys
            and value-types specified by this dictionary. Not all keys need be
            present in the imported labels
        -   ``None``. In this case, the importer makes no guarantees about the
            labels that it may return
        """
        return fo.Keypoints
    def setup(self):
        self._labels = [f for f in os.listdir(self.label_dir) if f.endswith(".txt")]
        self._labels = self._preprocess_list(self._labels)

    def close(self, *args):
        pass

# Usage
data_path = "./datasets/train/images"
labels_path = "./datasets/train/labels"

dataset = fo.Dataset.from_importer(
    YOLOv8PoseDatasetImporter(data_path, labels_path)
)

session = fo.launch_app(dataset)
session.wait()


