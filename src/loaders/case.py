from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from tqdm import tqdm
from torch.utils import data
from pathlib import Path
from typing import Literal, Union, Iterable
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import neurokit2 as nk
import os 
import re


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "case"
RAW_DATA_DIR = DATA_DIR / "case_interpolated"
CACHED_DATA_DIR = DATA_DIR / "cache"
DEFAULT_SUBJECTS_ARR = np.arange(1, 31)
DEFAULT_VIDEOS_ARR = np.arange(1, 9)


class CASERegression(data.Dataset):
    def __init__(
            self,
            category: Literal["arousal", "valence"],
            signals: list[str] = ['ecg'], 
            cached_data_dir: Union[str, Path] = CACHED_DATA_DIR,
            root: Union[str, Path] = RAW_DATA_DIR,
            rng_seed: int = 42,
            sr: int = 100,
            window_len: float = 10.0,
            window_shift: float = 5.0,
            args: dict = None
        ) -> None:
        """"""
        # save init params
        self.label_type = category
        self.signals = args.streams if args is not None else signals
        self.cached_data_dir = Path(cached_data_dir)
        self.rng_seed = rng_seed
        root = Path(root)
        if root.name != "case_interpolated":
            root = root / "case_interpolated"
        self.raw_data_dir = root
        self.target_physio_fs = sr
        self.window_len = window_len
        self.window_shift = window_shift
        self.subject_agnostic = args.subject_agnostic if args is not None else True
        # check if cached data
        if not self.check_cached_dir():
            # load and prepare data if no chache found
            # prepare_dataset - load all data, divide based on subject and video info, cut to same len, save in one file per signal
            print("No cached data found - preparing the dataset.")
            self.case_reader_object = _CASEReader(self.raw_data_dir / "physiological", self.raw_data_dir / "annotations")
            (self.physiology_time, self.annotations_time), (self.physiology, self.annotations), (self.subjects, self.videos) = self.load_dataset()
            for annotation_dim, annotations_data in self.annotations.items():
                self.annotations[annotation_dim] = self.normalize_annotations(annotations_data) 
            self.num_windows = num_windows
            windows_idxs, num_windows = self.make_window_index(self.physiology_time, self.window_len, window_shift)
            for signal_name, physio_signal in self.physiology.items():
                # get only data for current split
                # assign physiology based on specified windows
                physio_signal = self.assign_physiology(physio_signal, windows_idxs, signal_name)
                # if skt do not standardize - for skt we only have features, not time-series
                if signal_name != 'skt':
                    physio_signal = self.standardize_physio(physio_signal)
                # replace all physio with physio from current split
                self.physiology[signal_name] = physio_signal
            self.annotations = self.assign_annotations(self.annotations, self.annotations_time, self.physiology_time[windows_idxs])
            self.physiology, self.annotations, self.subjects, self.videos = self.extract_separate_windows(self.physiology, self.annotations, self.subjects, self.videos)
            self.save_to_cache()
        # load cached data
        (self.physiology_time, self.annotations_time), (self.physiology, self.annotations), (self.subjects, self.videos) = self.load_cached_data()
        # normalize annotations to a 0-1 range (minmax normalization)
        # make windows based on window length and window shift
        

        # for each physio signal
        
        # assign annotations for specified windows and ids
        # replace all subjects array with only subjects for current split

    def check_cached_dir(self):
        search_names = {'time', 'arousal', 'valence', 'subjects', 'videos', *self.signals}
        for f_path in self.cached_data_dir.glob("*.npy"):
            f_name = f_path.name[:-4]
            lookup_name = f_name.split('-')[-1]
            if lookup_name in search_names:
                search_names.remove(lookup_name)
        return len(search_names) == 0
            

    def load_dataset(self):
        "Prepare the dataset"
        # make cached data dir
        self.cached_data_dir.mkdir(parents=True, exist_ok=True)
        # prepare temp data storage
        physiology_time = None
        annotations_time = None
        physiology = dict()
        annotations = dict()
        subjects = list()
        videos = list()
        # iterate over initially processed data
        for vid_physiology, vid_annotations, (subject, vid_id) in self.case_reader_object.processed_data_iter():
            subjects.append(subject)
            videos.append(vid_id)
            for col_name, col_data in vid_physiology.items():
                if col_name not in self.signals and col_name != 'time':
                    continue
                physiology.setdefault(col_name, list())
                physiology[col_name].append(col_data.to_numpy())
            for col_name, col_data in vid_annotations.items():
                annotations.setdefault(col_name, list())
                annotations[col_name].append(col_data.to_numpy())
        for signal_name, signal_list in physiology.items():
            signal_array = np.stack(signal_list)
            if signal_name == "time":
                assert not np.any(signal_array - signal_array[0]), "Sth went wrong when reducing time in physio signals"
                physiology_time = signal_array[0]
                continue
            physiology[signal_name] = signal_array
        for annotation_dim, annotations_list in annotations.items():
            annotation_array = np.stack(annotations_list)
            if annotation_dim == "time":
                assert not np.any(annotation_array - annotation_array[0]), "Sth went wrong when reducing time in annotations"
                annotations_time = annotation_array[0]
                continue
            annotations[annotation_dim] = annotation_array
        del physiology['time']
        del annotations['time']
        subjects = np.stack(subjects)
        videos = np.stack(videos)
        return (physiology_time, annotations_time), (physiology, annotations), (subjects, videos)     

    def save_to_cache(self):
        np.save(self.cached_data_dir / f"physio-time.npy", self.physiology_time)
        np.save(self.cached_data_dir / f"annotations-time.npy", self.annotations_time)
        for signal_name, signal_array in self.physiology.items():
            np.save(self.cached_data_dir / f"physio-{signal_name}.npy", signal_array)
        for annotation_dim, annotation_array in self.annotations.items():
            np.save(self.cached_data_dir / f"annotations-{annotation_dim}.npy", annotation_array)
        np.save(self.cached_data_dir / "subjects.npy", self.subjects)
        np.save(self.cached_data_dir / "videos.npy", self.videos)  

    def prepare_cv(self):
        data_random_idxs, kfold_ids = self.setup_lkso_kfold(
            n_splits=self.num_folds
        )
        return data_random_idxs, kfold_ids

    # def get_validation_and_test_idxs(self, idxs):
    #     """
    #     Generate validation idxs so that validation dataset contains half the samples from the test set (subject-agnostic scenario)
    #     or one sample from each subject (subject-dependent scenario)
    #     """
    #     rng = np.random.default_rng(seed=self.rng_seed)
    #     permutation = rng.permutation(idxs)
    #     validation_ids = permutation[:len(idxs)//2]
    #     test_ids = permutation[len(idxs)//2:]
    #     return validation_ids, test_ids
    
    # def get_random_idxs(self):
    #     rng = np.random.default_rng(seed=self.rng_seed)
    #     idxs = np.arange(len(self.subjects))
    #     idxs = rng.permutation(idxs)
    #     return idxs

    # def setup_lkso_kfold(self, n_splits=5):
    #     idxs = self.get_random_idxs()
    #     # make kfold object for splitting data
    #     kfold = GroupKFold(n_splits=n_splits)
    #     return idxs, list(kfold.split(idxs, groups=self.subjects[idxs].astype(int)))

    def load_cached_data(self):
        physiology = dict()
        annotations = dict()
        physiology_time = np.load(self.cached_data_dir / "physio-time.npy")
        annotations_time = np.load(self.cached_data_dir / "annotations-time.npy")
        for signal in self.signals:
            physiology.setdefault(signal, np.load(self.cached_data_dir / f"physio-{signal}.npy"))
        annotations = np.load(self.cached_data_dir / f"annotations-{self.label_type}.npy")
        subjects = np.load(self.cached_data_dir / "subjects.npy")
        videos = np.load(self.cached_data_dir / "videos.npy")
        return (physiology_time, annotations_time), (physiology, annotations), (subjects, videos)

    def standardize_physio(self, data, mean=None, std=None, return_stats=False):
        compute_axis = (1,2)
        mean = mean if mean is not None else np.mean(data, axis=compute_axis)
        std = std if std is not None else np.std(data, axis=compute_axis)
        mean = mean[:, None, None]
        std = std[:, None, None]
        data = (data - mean) / (std + 1e-5)
        if return_stats:
            return data, mean, std
        return data

    def normalize_annotations(self, annotations, inplace=False):
        # annotations range is 0.5 - 9.5
        # min - max normalization
        if not inplace:
            annotations = deepcopy(annotations)
        annotations = (annotations - 0.5) / (9.5 - 0.5)
        return annotations

    def filter_ecg(self, ecg, sr=None):
        # filtering
        sr = sr or self.target_physio_fs
        ecg = nk.ecg_clean(ecg, sampling_rate=sr)
        # ecg = self.standardize_data(ecg)
        return ecg
    
    def filter_gsr(self, gsr, sr=None):
        # filtering
        sr = sr or self.target_physio_fs
        gsr = nk.eda_phasic(nk.standardize(gsr), sampling_rate=sr)["EDA_Phasic"].values
        # gsr = self.standardize_data(gsr)
        return gsr
    
    def filter_rsp(self, rsp, sr=None):
        # filtering
        sr = sr or self.target_physio_fs
        rsp = nk.rsp_clean(rsp, sampling_rate=sr)
        # rsp = self.standardize_data(rsp)
        return rsp
    
    def compute_skt_features(self, skt, sr=None):
        # filtering
        features_arr = np.zeros(4)
        features_arr[0] = np.mean(skt)
        features_arr[1] = np.std(skt)
        skt_diff = np.diff(skt)
        features_arr[2] = np.max(skt_diff)
        features_arr[3] = np.std(skt_diff)
        return features_arr

    def make_window_index(
        self, index, window_size, window_shift, discard_edge_windows=False
    ):
        "Split data into windows of fixed length, so that annotations and physiology match"
        # ensure that annotations will always have corresponding physiology
        window_size *= self.target_physio_fs
        window_shift *= self.target_physio_fs
        window_size = int(window_size)
        num_windows = int((len(index) - window_size) / window_shift) + 1
        windowed_index = np.arange(window_size)[None, :] + int(window_shift)*np.arange(num_windows)[:, None]
        if discard_edge_windows:
            windowed_index = windowed_index[1:-1]
            num_windows = num_windows - 2
        return windowed_index, num_windows

    def assign_annotations(self, annotations, annotations_times, physiology_widows_times, center_annotations=True):
        for annotation_dim, annotation_array in annotations.items():
            windows_annotations = list()
            annotations_index = np.arange(len(annotations_times))
            for window_times in physiology_widows_times:
                window_annotation = annotations_index[(annotations_times >= window_times[0]) & (annotations_times <= window_times[-1])]
                mid_annotation = annotation_array[:, window_annotation[len(window_annotation)//2]]
                windows_annotations.append(mid_annotation)
            annotations[annotation_dim] = np.stack(windows_annotations, -1)
        return annotations
    
    def assign_physiology(self, physiology, physiology_widows_idxs, signal_name):
        processing_func = self.filtering_methods[signal_name]
        physiology = np.apply_along_axis(processing_func, -1, physiology[:, physiology_widows_idxs])
        return physiology

    def extract_separate_windows(self, physiology, annotations, subjects, videos):
        subjects = np.concatenate(subjects[:, None] + np.zeros(self.num_windows)[None, :])
        videos = np.concatenate(videos[:, None] + np.zeros(self.num_windows)[None, :])
        for annotation_dim, annotation_array in annotations.items():
            annotations[annotation_dim] = np.concatenate(annotation_array)
        for signal_name, physio in physiology.items():
            physiology[signal_name] = np.concatenate(physio)
        return physiology, annotations, subjects, videos
    
    @property
    def filtering_methods(self):
        return {
            'ecg': self.filter_ecg,
            'gsr': self.filter_gsr,
            'rsp': self.filter_rsp,
            'skt': self.compute_skt_features,
        }
    
    @property
    def names(self):
        if not self.subject_agnostic:
            return 10*self.subjects + self.videos
        return copy(self.subjects)
    
    @property
    def labels(self):
        return self.annotations

    def __len__(self):
        # TODO rethink it - maybe len of some other array
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.signals == ['ecg']:
            ret_signals =  self.physiology['ecg'][idx]
        else:
            ret_signals = {
                signal: self.physiology[signal][idx] for signal in self.signals
            }
        ret_annotations = self.annotations[idx]
        return ret_signals, ret_annotations, 0


class CASEClassification(CASERegression):
    def __init__(
            self, 
            category: Literal['arousal', 'valence'],
            signals: list[str] = ['ecg'], 
            cached_data_dir: Union[str, Path] = CACHED_DATA_DIR,
            root: Union[str, Path] = RAW_DATA_DIR,
            rng_seed: int = 42, 
            sr: int = 100, 
            window_len: float = 10, 
            window_shift: float = 5,
            args: dict = None
        ) -> None:
        super().__init__(category, signals, cached_data_dir, root, rng_seed, sr, window_len, window_shift, args)
        self.annotations, new_indexes = self.assign_classes(self.annotations)
        for signal_name, signal in self.physiology.items():
            self.physiology[signal_name] = signal[new_indexes]
        self.subjects, self.videos = self.subjects[new_indexes], self.videos[new_indexes]
    
    def assign_classes(self, annotations, high_boundary=0.5, low_boundary=None):
        # everything above or equal this value will have class 1 assigned
        # everything below or equal this value will have class 0 assigned
        # if conditioning on only one value (no "dead-zone"), provide low_boundary > high_boundary
        if low_boundary is None:
            low_boundary = np.inf
        # prepare index storage
        new_indexes = np.zeros(annotations.shape) + 2
        # match boundary scores
        new_indexes[annotations < high_boundary] -= 1
        new_indexes[annotations > low_boundary] -= 1
        # convert to bool = those that are between low and high boundary (dead zone) are False
        new_indexes = new_indexes.astype(bool)
        # remove the dead zone from annotations
        selected_annotations = annotations[new_indexes]
        # assign classes to selected annotations
        new_annotations = (selected_annotations >= high_boundary).astype(int)
        # return annotations
        return new_annotations, new_indexes


class _CASEReader:
    def __init__(self, physiology_dir, annotations_dir, videos=None, subjects=None, cut_data = True, cut_length = 118, cut_from_end = True, resample_physio = True, target_physio_fs = 100) -> None:
        self.videos = np.array(videos) or DEFAULT_VIDEOS_ARR
        self.subjects = np.array(subjects) or DEFAULT_SUBJECTS_ARR
        self.subjects_set = set(self.subjects)
        self.data_paths, self.sub_pathsid_map = self.generate_data_paths(
            physiology_dir, annotations_dir
        )
        self.if_cut_data = cut_data
        self.if_resample_physio = resample_physio
        self.target_physio_fs = target_physio_fs
        self.cut_length = cut_length
        self.cut_from_end = cut_from_end

    @staticmethod
    def get_subject_num(f_name):
        """
        Get subject (participant) id from file name.
        Args:
            f_name (str | Path): file name containing subject number
        
        """
        if isinstance(f_name, Path):
            f_name = f_name.name
        # search for numbers in file name. Numbers correspond to participants
        res = re.search(r"\d+", f_name)
        # if res is None, then there was no number in f_name
        if res is None:
            return -1
        # if res is not None, extract the number
        return int(res.group())
    
    def generate_data_paths(self, physiology_dir, annotations_dir):
        "Generate tuples of (subject annotations, subject physiology)"
        paths_list = list()
        path_mapping = dict()
        # iterate over files in annotations and physiology dirs
        for physio_path, annot_path in zip(
            sorted(Path(physiology_dir).iterdir(), key=self.get_subject_num),
            sorted(Path(annotations_dir).iterdir(), key=self.get_subject_num),
        ):
            # ensure that both files are for the same participant
            # if names do not match
            if annot_path.name != physio_path.name:
                # ignore if files are readme files
                if ("readme" in annot_path.name.lower()) and ("readme" in annot_path.name.lower()):
                    continue
                # raise exception if not readme files (files order do not match)
                raise Exception("Mismatched order")
            subject_id = self.get_subject_num(annot_path.name)
            if subject_id not in self.subjects_set:
                continue
            # add (annotations_path, physiology_path) for current participant
            paths_list.append(
                (
                    physio_path,
                    annot_path,
                )
            )
            path_mapping.setdefault(subject_id, len(paths_list) - 1)
        return paths_list, path_mapping
    
    def load_data(self, path):
        data_load = pd.read_csv(path)
        data_load.rename(columns={"jstime": "time", "daqtime": "time"}, inplace=True)
        return data_load
    
    def get_subject_data(self, subject_id):
        physiology_path, annotations_path = self.data_paths[
            self.sub_pathsid_map.get(subject_id)
        ]
        physiology, annotations = self.load_data(physiology_path), self.load_data(annotations_path)
        return physiology, annotations

    def iterate_subjects_data(self):
        for subject in self.subjects:
            yield (subject, self.get_subject_data(subject))

    def get_video_data(self, subject_data, video, fix_mismatched=True):
        "Get subject data for the specified video"
        # unpack subject's data
        subject_physiology, subject_annotations = subject_data
        # choose data only for the specified video
        video_annotations = subject_annotations[
            subject_annotations["video"] == video
        ].copy()
        video_physiology = subject_physiology[
            subject_physiology["video"] == video
        ].copy()
        # remove video col from dataframes
        video_physiology.drop(columns=['video'], inplace=True)
        video_annotations.drop(columns=['video'], inplace=True)
        # ensure that physiology and annotations timestamps match
        if fix_mismatched:
            self.trim_to_same_time(video_physiology, video_annotations)
        # get starting time
        start_time = video_physiology["time"].iloc[0]
        # ensure that time is the first column
        video_annotations.insert(0, "time", video_annotations.pop("time") - start_time)
        video_physiology.insert(0, "time", video_physiology.pop("time") - start_time)
        # reset time index, so it starts from 0. Drop old index
        video_annotations.reset_index(drop=True, inplace=True)
        video_physiology.reset_index(drop=True, inplace=True)
        # return video data
        return video_physiology, video_annotations
    
    def trim_to_same_time(self, physiology, annotations):
        annotations.drop(
            axis="index",
            # get rows of annotations, where time < time at the beginning of physiology and rows of annotations, where time > time at the end of physiology
            index=annotations[
                (annotations["time"] < physiology["time"].iloc[0])
                | (annotations["time"] > physiology["time"].iloc[-1])
            ].index,
            inplace=True,
        )
        common_time = (physiology["time"] >= annotations["time"].iloc[0]) & (
            physiology["time"] <= annotations["time"].iloc[-1]
        )
        physiology.drop(
            axis="index",
            # get rows of annotations, where time < time at the beginning of physiology and rows of annotations, where time > time at the end of physiology
            index=physiology[~common_time].index,
            inplace=True,
        )
        # return annotations, physiology
    
    def _resample_ts(self, signal, signal_fs=1000):
        """
        """
        res = resample_poly(signal, self.target_physio_fs, signal_fs)
        return res
    
    def resample_data_simple(
        self, data, signal_fs=1000, cut_data='front'
    ):
        fs_rate = signal_fs / self.target_physio_fs
        data_overhead = len(data) % self.target_physio_fs
        resample_signal_len = int((len(data) - data_overhead)) * self.target_physio_fs / signal_fs
        time = data['time'].to_numpy()
        if cut_data == 'front':
            data = data.iloc[data_overhead:]
            new_time = np.arange(1, resample_signal_len + 1)*fs_rate + time[0]
        elif cut_data == 'end':
            data = data.iloc[:-data_overhead]
            new_time = np.arange(0, resample_signal_len)*fs_rate + time[0]
        else:
            print(f"Wrong value: cut_data == {cut_data}")
            return
        signal = np.array(data.drop(columns=['time']))
        signal_resample = self._resample_ts(signal)
        new_time = np.reshape(new_time, (-1,1))
        data_resample = np.concatenate([new_time, signal_resample], axis=1)
        data_resample = pd.DataFrame(data_resample, columns=data.columns)
        return data_resample

    def iterate_videos_data(self, data):
        for video in self.videos:
            yield (video, self.get_video_data(data, video))

    def cut_data(self, data, wanted_length, time_scale=1000, from_end=False):
        wanted_length = wanted_length * time_scale
        if from_end:
            wanted_start_time = data['time'].iloc[-1] - wanted_length
            ret_data = data[data['time'] >= wanted_start_time].reset_index(drop=True)
            return ret_data
        return data[data['time'] <= wanted_length]

    def processed_data_iter(self):
        for subject, subject_data in tqdm(self.iterate_subjects_data()):
            for video, video_data in self.iterate_videos_data(subject_data):
                video_physiology, video_annotations = video_data
                # cut data to same length
                if self.if_cut_data:
                    video_annotations = self.cut_data(video_annotations, self.cut_length, from_end=self.cut_from_end)
                    video_physiology = self.cut_data(video_physiology, self.cut_length, from_end=self.cut_from_end)
                    video_annotations['time'] = video_annotations['time'] - video_physiology['time'].iloc[0]
                    video_physiology['time'] = video_physiology['time'] - video_physiology['time'].iloc[0]
                if self.if_resample_physio:
                    video_physiology = self.resample_data_simple(video_physiology)
                yield video_physiology, video_annotations, (subject, video)
