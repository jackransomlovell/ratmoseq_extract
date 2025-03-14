import cv2
import h5py
import pickle
import pickle
import joblib
import panel as pn
import numpy as np
import holoviews as hv
from pathlib import Path
from sklearn.svm import SVC
from ruamel.yaml import YAML
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from ratmoseq_extract.proc import clean_frames, min_max_scale

yaml = YAML(typ="safe", pure=True)


def get_flips(frames, flip_path):
    """Get flip predictions for frames using a trained classifier.
    
    Args:
        frames (np.ndarray): Input frames to classify
        flip_path (str): Path to saved classifier pipeline
        
    Returns:
        np.ndarray: Binary predictions indicating which frames should be flipped
    """
    try:
        # Load the classifier pipeline with custom persistence
        with open(flip_path, 'rb') as f:
            flip = pickle.load(f)
        # Get predictions
        preds = flip.predict(frames)
        return preds
    except Exception as e:
        print(f"Error loading flip classifier: {e}")
        # Return no flips if classifier fails
        return np.zeros(len(frames), dtype=bool)

def apply_flips(h5, flip_classif, smoothing=0, save=True):
    # apply flips
    with h5py.File(h5, 'a') as f:
        frames = f['frames'][:]
        flips = get_flips(
            frames, flip_classif
        )
        flip_inds = np.where(flips)
        if save:
            f['frames'][flip_inds] = np.rot90(
                frames[flip_inds], k=2, axes=(1, 2)
            )
    return flip_inds

@dataclass
class CleanParameters:
    prefilter_space: tuple = (5,)
    strel_tail: Tuple[int, int] = (9, 9)
    iters_tail: Optional[int] = 1
    height_threshold: int = 5

def create_training_dataset(
    data_index_path: str, clean_parameters: Optional[CleanParameters] = None
) -> str:
    np.random.seed(0)
    data_index_path = Path(data_index_path)
    out_path = data_index_path.with_name("training_data.npz")
    if clean_parameters is None:
        clean_parameters = CleanParameters()

    # load trainingdata index
    with open(data_index_path, "rb") as f:
        session_paths, data_index = pickle.load(f)

    # load frames
    frames = []
    for k, v in data_index.items():
        with h5py.File(session_paths[k], "r") as h5f:
            for left, _slice in v:
                frames_subset = h5f["frames"][_slice]
                if left:
                    frames_subset = np.rot90(frames_subset, 2, axes=(1, 2))
                frames.append(frames_subset)
    frames = np.concatenate(frames, axis=0)
    # rotate frames
    frames = np.concatenate((frames, np.rot90(frames, 2, axes=(1, 2))), axis=0)

    flipped = np.zeros((len(frames),), dtype=np.uint16)
    flipped[len(frames) // 2 :] = 1

    # add some randomly shifted frames
    shifts = np.random.randint(-5, 5, size=(len(frames), 2))
    shifted_frames = np.array(
        [np.roll(f, tuple(s), axis=(0, 1)) for f, s in zip(frames, shifts)]
    ).astype(np.uint16)

    # remove noise from frames
    cleaned_frames = clean_frames(
        frames.astype(np.uint16),
        clean_parameters.prefilter_space,
        strel_tail=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, clean_parameters.strel_tail
        ),
        iters_tail=clean_parameters.iters_tail
    )
    frames = np.concatenate((frames, shifted_frames, cleaned_frames), axis=0)
    print(
        f"Training data shape: {frames.shape}; memory usage: {frames.nbytes / 1e9 * 4:0.2f} GB"
    )

    flipped = np.concatenate((flipped, flipped, flipped), axis=0)

    np.savez(out_path, frames=frames, flipped=flipped)
    return out_path


def flatten(array: np.ndarray) -> np.ndarray:
    return array.reshape(len(array), -1)


def batch_apply_pca(frames: np.ndarray, pca: PCA, batch_size: int = 1000) -> np.ndarray:
    output = []
    if len(frames) < batch_size:
        return pca.transform(flatten(frames)).astype(np.float32)

    for arr in np.array_split(frames, len(frames) // batch_size):
        output.append(pca.transform(flatten(arr)).astype(np.float32))
    return np.concatenate(output, axis=0).astype(np.float32)


from sklearn.base import BaseEstimator, TransformerMixin
# Custom transformer to reshape 3D data to 2D in batches
class BatchPCA(BaseEstimator, TransformerMixin):
    def __init__(self, pca, batch_size=1000):
        self.pca = pca
        self.batch_size = batch_size

    def fit(self, X, y=None):
        # Flatten and fit PCA on the data in batches
        n_samples, _, _ = X.shape
        self.pca.fit(flatten(X[:-n_samples // 3]))  # Fit PCA on the flattened data
        return self

    def transform(self, X):
        # Transform data in batches
        n_samples, _, _ = X.shape
        output = []

        # Process in batches to avoid memory overload
        for i in range(0, n_samples, self.batch_size):
            transformed_batch = self.pca.transform(flatten(X[i:i + self.batch_size]))
            output.append(transformed_batch)

        return np.concatenate(output, axis=0)

def train_classifier(
    data_path: str,
    classifier: str = "SVM",
    n_components: int = 20,
):
    """Train a classifier to predict the orientation of a mouse.
    Parameters:
        data_path (str): Path to the training data numpy file.
        classifier (str): Classifier to use. Either 'SVM' or 'RF'.
        n_components (int): Number of components to keep in PCA."""
    data = np.load(data_path)
    frames = data["frames"]
    flipped = data["flipped"]

    print("Fitting PCA")
    pca = PCA(n_components=20)
    batch_size = 100
    
    pipeline = make_pipeline(
        BatchPCA(pca, batch_size=batch_size),  # Apply PCA in batches
        StandardScaler(),
        RandomForestClassifier(n_estimators=150)
    )

    print("Running cross-validation")
    accuracy = cross_val_score(
        pipeline, frames, flipped, cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    print(f"Held-out model accuracy: {accuracy.mean()}")

    print("Final fitting step")
    return pipeline.fit(frames, flipped)


def save_classifier(clf_pipeline, out_path: str):
    """Save the trained classifier pipeline.
    
    Args:
        clf_pipeline: Trained sklearn pipeline
        out_path (str): Path to save the classifier
    """
    with open(out_path, 'wb') as f:
        pickle.dump(clf_pipeline, f)
    print(f"Classifier saved to {out_path}")


def _extraction_complete(file_path: Path):
    config = yaml.load(file_path.read_text())
    return config["complete"]

def _check_h5(file, name):
    with h5py.File(file, 'r') as f:
        return name in f

def _find_extractions(data_path: str, frames_name: str):
    files = Path(data_path).glob("**/results_00.h5")
    files = sorted(f for f in files if _extraction_complete(f.with_suffix(".yaml")))
    files = [f for f in files if _check_h5(f, frames_name)]
    if len(set([f.name for f in files])) < len(files):
        files = {f.parents[1].name + '/' + f.name: f for f in files}
    else:
        files = {f.parents[1].name: f for f in files}
    
    return files


class FlipClassifierWidget:
    def __init__(self, data_path: str, frames_name: str = "frames"):
        self.data_path = Path(data_path)
        self.frames_name = frames_name
        self.sessions = _find_extractions(data_path, frames_name)
        # self.selected_frame_ranges_dict = {k: [] for k in self.path_dict}
        self.selected_frame_ranges_dict = defaultdict(list)
        self.curr_total_selected_frames = 0

        self.session_select_dropdown = pn.widgets.Select(
            options=list(self.sessions), name="Session", value=list(self.sessions)[1]
        )
        self.frame_num_slider = pn.widgets.IntSlider(
            name="Current Frame", start=0, end=1000, step=1, value=1
        )
        self.start_button = pn.widgets.Button(name="Start Range", button_type="primary")
        self.face_left_button = pn.widgets.Button(
            name="Facing Left", button_type="success", width=140, visible=False
        )
        self.face_right_button = pn.widgets.Button(
            name="Facing Right", button_type="success", width=140, visible=False
        )
        self.selected_ranges = pn.widgets.MultiSelect(
            name="Selected Ranges", options=[]
        )
        self.delete_selection_button = pn.widgets.Button(
            name="Delete Selection", button_type="danger"
        )
        self.curr_total_label = pn.pane.Markdown(
            f"Current Total Selected Frames: {self.curr_total_selected_frames}"
        )

        self.facing_info = pn.pane.Markdown(
            "To finish selection, click on direction the animal is facing",
            visible=False,
        )
        self.facing_row = pn.Row(self.face_left_button, self.face_right_button)
        self.range_box = pn.Column(
            self.curr_total_label,
            pn.pane.Markdown("#### Selected Correct Frame Ranges"),
            self.selected_ranges,
            self.delete_selection_button,
        )

        self.forward_button = pn.widgets.Button(
            name="Forward", button_type="primary", width=142
        )
        self.backward_button = pn.widgets.Button(
            name="Backward", button_type="primary", width=142
        )

        self.frame_advancer_row = pn.Row(self.backward_button, self.forward_button)

        self.widgets = pn.Column(
            self.session_select_dropdown,
            self.frame_num_slider,
            self.start_button,
            self.frame_advancer_row,
            self.facing_info,
            self.facing_row,
            self.range_box,
            width=325,
        )

        self.frame_display = hv.DynamicMap(
            self.display_frame, streams=[self.frame_num_slider.param.value]
        ).opts(
            frame_width=400, frame_height=400, aspect="equal", xlim=(0, 1), ylim=(0, 1)
        )

        self.start_button.on_click(self.start_stop_frame_range)
        self.face_left_button.on_click(
            lambda event: self.facing_range_callback(event, True)
        )
        self.face_right_button.on_click(
            lambda event: self.facing_range_callback(event, False)
        )
        self.delete_selection_button.on_click(self.on_delete_selection_clicked)

        self.forward_button.on_click(self.advance_frame)
        self.backward_button.on_click(self.rewind_frame)

        self.session_select_dropdown.param.watch(self.changed_selected_session, "value")
        self.session_select_dropdown.value = list(self.sessions)[0]

    def display_frame(self, value):
        if hasattr(self, "frames"):
            frame = self.frames[value]
        else:
            frame = None
        # set bounds for the image
        return hv.Image(frame, bounds=(0, 0, 1, 1)).opts(cmap="cubehelix")

    def advance_frame(self, event):
        if self.frame_num_slider.value < self.frame_num_slider.end:
            self.frame_num_slider.value += 1

    def rewind_frame(self, event):
        if self.frame_num_slider.value > 0:
            self.frame_num_slider.value -= 1

    def start_stop_frame_range(self, event):
        if self.start_button.name == "Start Range":
            self.start = self.frame_num_slider.value
            self.start_button.name = "Cancel Select"
            self.start_button.button_type = "danger"
            self.face_left_button.visible = True
            self.face_right_button.visible = True
            self.facing_info.visible = True
            self.facing_info.object = f"To finish selection, click on direction the animal is facing. Start: {self.start}"
        else:
            self.start_button.name = "Start Range"
            self.start_button.button_type = "primary"
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False

    def facing_range_callback(self, event, left):
        self.stop = self.frame_num_slider.value
        if self.stop > self.start:
            self.update_state_on_selected_range(left)
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False
            self.start_button.name = "Start Range"
            self.start_button.button_type = "primary"

    def update_state_on_selected_range(self, left):
        selected_range = range(self.start, self.stop)

        beginning = "L" if left else "R"
        display_selected_range = (
            f"{beginning} - {selected_range} - {self.session_select_dropdown.value}"
        )

        self.curr_total_selected_frames += len(selected_range)
        self.curr_total_label.object = (
            f"Current Total Selected Frames: {self.curr_total_selected_frames}"
        )

        self.selected_frame_ranges_dict[self.session_select_dropdown.value].append(
            (left, selected_range)
        )

        self.selected_ranges.options = self.selected_ranges.options + [
            display_selected_range
        ]
        self.selected_ranges.value = []

        self.save_frame_ranges()

    def on_delete_selection_clicked(self, event):
        selected_range = self.selected_ranges.value
        if selected_range:
            vals = selected_range[0].split(" - ")
            delete_key = vals[2]
            direction = vals[0] == "L"
            range_to_delete = eval(vals[1])

            to_drop = (direction, range_to_delete)
            self.selected_frame_ranges_dict[delete_key].remove(to_drop)

            self.curr_total_selected_frames -= len(range_to_delete)
            self.curr_total_label.object = (
                f"Current Total Selected Frames: {self.curr_total_selected_frames}"
            )

            l = list(self.selected_ranges.options)
            l.remove(selected_range[0])

            self.selected_ranges.options = l
            self.selected_ranges.value = []

    def changed_selected_session(self, event):
        if self.start_button.name == "Cancel Select":
            self.start_button.name = "Start Range"
            self.start_button.button_type = "primary"

        self.frame_num_slider.value = 1
        self.widgets.loading = True

        with h5py.File(self.sessions[event.new], mode="r") as f:
            self.frame_num_slider.end = f[self.frames_name].shape[0] - 1
            self.frames = f[self.frames_name][()]

        self.widgets.loading = False

        self.frame_num_slider.value = 0

    def show(self):
        return pn.Row(self.widgets, self.frame_display)

    def save_frame_ranges(self):
        with open(self.training_data_path, "wb") as f:
            pickle.dump((self.sessions, dict(self.selected_frame_ranges_dict)), f)

    @property
    def training_data_path(self):
        return str(self.data_path.parent / "flip-training-frame-ranges.p")
