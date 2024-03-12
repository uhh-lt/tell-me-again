import glob
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from platformdirs import user_cache_dir
from tqdm import tqdm

try:
    import torch
except ImportError:
    import numpy
import itertools
import json
import random
import zipfile

import requests
from . import util

VERSION = "v1"
DOWNLOAD_URL = f"https://ltdata1.informatik.uni-hamburg.de/tell_me_again_{VERSION}.zip"

MAX_RETRIES = 0
TRANSLATION_SCORES = {
    "en": 100,
    "fr": 68.1,
    "de": 67.4,
    "it": 61.2,
    "es": 59.1,
}


def download_dataset(url=DOWNLOAD_URL, retry_count: int = 0) -> Path:
    cache_dir = user_cache_dir("tell_me_again", "uhh-lt")
    out_file_path = Path(cache_dir) / "data.zip"
    if os.path.exists(out_file_path):
        zip_file = zipfile.ZipFile(out_file_path)
        try:
            version_file = zip_file.open("version.txt", "r")
        except:
            version_file = None
            # raise ValueError("Invalid archive (no version information present), delete the file in", cache_dir, "and run again.")
        if (
            version_file is not None
            and (actual_version := next(version_file).strip().decode("utf-8"))
            == VERSION
        ):
            return out_file_path
        else:
            print(
                f"Old version of the tell_me_again dataset found (desired {VERSION} != actual version {actual_version}), please manually remove the files in {cache_dir}"
            )
            raise ValueError("Old dataset version found.")
    if retry_count > MAX_RETRIES:
        raise ValueError("Download Failed")
    os.makedirs(cache_dir, exist_ok=True)
    util.download(url, out_file_path)
    return out_file_path


def get_genres(wikidata_dict):
    genres = wikidata_dict["claims"].get("P136", [])
    genre_ids = [
        e["mainsnak"]["datavalue"]["value"]["id"]
        for e in genres
        if e["mainsnak"]["snaktype"] != "novalue"
    ]
    return genre_ids


@dataclass
class Story:
    wikidata_id: str
    description: str
    titles: Dict[str, str]
    title: str
    summaries_original: Dict[str, str]
    summaries_translated: Dict[str, str]
    summaries_anonymized: Optional[Dict[str, str]]
    similarities: any
    similarities_labels: List[str]
    num_sentences: Dict[str, int]
    sentences: Dict[str, List[str]]
    genres: List[str]

    @classmethod
    def from_dict(cls, data, wikidata_data=None):
        if wikidata_data is not None:
            genres = get_genres(wikidata_data)
        else:
            genres = []
        sentences = {
            k: s["sentences"]
            for k, s in data.get("en_translated_summaries", {}).items()
        }
        num_sentences = {
            k: len(s["sentences"])
            for k, s in data.get("en_translated_summaries", {}).items()
        }
        if "en" in (sents := data.get("split_into_sents")):
            num_sentences.update({"en": len(sents["en"])})
            sentences.update({"en": sents["en"]})
        if "torch" in sys.modules:
            sims = torch.tensor(data.get("similarity", {}).get("similarities", []))
        else:
            sims = numpy.array(data.get("similarity", {}).get("similarities", []))
        return cls(
            wikidata_id=data["wikidata_id"],
            titles={
                k: (v or {}).get("value") for k, v in data.get("titles", {}).items()
            },
            title=data["title"],
            description=data["description"],
            summaries_original=data["summaries"],
            summaries_translated={
                k: s["text"] for k, s in data.get("en_translated_summaries", {}).items()
            },
            similarities_labels=data.get("similarity", {}).get("indexes"),
            similarities=sims,
            summaries_anonymized=data.get("anonymized"),
            num_sentences=num_sentences,
            sentences=sentences,
            genres=genres,
        )

    def remove_duplicates(self, threshold=0.6):
        out = {}
        sorted_labels = sorted(
            self.similarities_labels, key=lambda x: TRANSLATION_SCORES[x]
        )
        sorted_similarities = [
            [
                v.item()
                for k, v in sorted(
                    zip(self.similarities_labels, sim),
                    key=lambda kv: TRANSLATION_SCORES[kv[0]],
                    reverse=True,
                )
            ]
            for sim in self.similarities
        ]
        for i, (lang, text) in enumerate(
            sorted(
                self.summaries_translated.items(),
                key=lambda kv: TRANSLATION_SCORES[kv[0]],
                reverse=True,
            )
        ):
            try:
                index = (sorted_labels or []).index(lang)
            except ValueError:
                print(lang)
                index = None
            try:
                max_value = max(sorted_similarities[index][:i])
            except ValueError:
                max_value = 0
            if index is not None and max_value > threshold:
                pass
            else:
                out[lang] = text
        return out

    def get_anonymized(self, min_sentences=0):
        return {
            lang: text
            for lang, text in self.summaries_anonymized.items()
            if self.num_sentences[lang] >= min_sentences
        }

    def get_all_summaries_en(self, max_similarity=0.6, min_sentences=0):
        en = self.summaries_original.get("en")
        summaries = []
        ids = []
        if en is not None:
            summaries.append(en)
            ids.append("en")
        no_dups = self.remove_duplicates()
        summaries += [e for e in no_dups.values()]
        ids += [e for e in no_dups.keys()]
        ids = [id_ for id_ in ids if self.num_sentences[id_] >= min_sentences]
        summaries = [
            s
            for (id_, s) in zip(ids, summaries)
            if self.num_sentences[id_] >= min_sentences
        ]
        return ids, summaries

    def __repr__(self):
        return f"<Story title='{self.title}' description='{self.description}'>"


class StoryDataset:
    def __init__(self, data_path: Optional[Path] = None, only_include=[], stories=None):
        if data_path is None:
            data_path = download_dataset()
        self.data_path = data_path
        self.stories = stories or {}
        if len(self.stories) > 0:
            return
        zip_file = zipfile.ZipFile(self.data_path)
        for file_name in tqdm(
            util.zip_glob(zip_file, "summaries/*/*.json"),
            desc="Loading summaries",
        ):
            wikidata_id = os.path.splitext(os.path.basename(file_name))[0]
            wikidata_data = json.load(
                zip_file.open(f"wikidata/{wikidata_id[:2]}/{wikidata_id}.json")
            )
            if len(only_include) > 0 and (wikidata_id not in only_include):
                continue
            else:
                self.stories[wikidata_id] = Story.from_dict(
                    json.load(zip_file.open(file_name)), wikidata_data
                )

    def __iter__(self):
        yield from self.stories.values()

    def __getitem__(self, i):
        return self.stories[i]

    def __len__(self):
        return len(self.stories)

    def perform_splits(self):
        split_ids = {}
        for split in ["train", "dev", "test"]:
            in_file = open(
                self.data_path / Path(split + "_stories.csv")
            )  # TODO: fix path
            split_ids[split] = [l.strip() for l in in_file.readlines()]
        train_stories = {k: self.stories[k] for k in split_ids["train"]}
        dev_stories = {k: self.stories[k] for k in split_ids["dev"]}
        test_stories = {k: self.stories[k] for k in split_ids["test"]}
        return {
            k: self.__class__(data_path=None, stories=s)
            for k, s in [
                ("train", train_stories),
                ("dev", dev_stories),
                ("test", test_stories),
            ]
        }

    def chaturvedi_like_split(self, use_anonymized: bool = False, seed=1337):
        target_length_count = {2: 235, 3: 20, 4: 10, 5: 1}
        randomizer = random.Random(seed)
        ids = list(self.stories.keys())
        randomizer.shuffle(ids)
        by_length = defaultdict(list)
        all_summaries = []
        all_summaries_test = []
        labels = []
        labels_test = []
        included = []
        for id_ in ids:
            if use_anonymized:
                summaries = self.stories[id_].summaries_anonymized.values()
            else:
                _, summaries = self.stories[id_].get_all_summaries_en()
            if len(by_length.get(len(summaries), [])) < target_length_count.get(
                len(summaries), 0
            ):
                in_test_set = [
                    True if randomizer.random() <= 0.8 else False
                    for _ in range(len(summaries))
                ]
                included.extend(in_test_set)
                test_summaries = [s for t, s in zip(in_test_set, summaries) if t]
                labels.extend([id_] * len(summaries))
                labels_test.extend([id_] * len(test_summaries))
                all_summaries.extend(summaries)
                all_summaries_test.extend(summaries)
                by_length[len(summaries)].append(summaries)
        return all_summaries, labels, included

    def stratified_split(self, label_dict, seed=2):
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=2, random_state=seed, shuffle=True)
        splits = list(
            splitter.split(list(label_dict.keys()), list(label_dict.values()))
        )
        ids = list(label_dict.keys())
        return [
            [(label_dict[ids[i]], self[ids[i]]) for i in split] for split in splits[0]
        ]

    def get_metadata_stats(self):
        book_count, movie_count, both_count = 0, 0, 0
        has_gutenberg = 0
        has_isbn = 0
        genre_counter = Counter()
        count = 0
        neither_count = 0
        for _, story in tqdm(self.stories.items()):
            data = json.load(
                open(
                    f"data/wikidata/{story.wikidata_id[1:3]}/{story.wikidata_id[1:]}.json"
                )
            )
            genres = data["claims"].get("P136", [])
            genre_ids = [
                e["mainsnak"]["datavalue"]["value"]["id"]
                for e in genres
                if e["mainsnak"]["snaktype"] != "novalue"
            ]
            gutenberg = data["claims"].get("P2034", [])
            gutenberg_ids = [
                e["mainsnak"]["datavalue"]["value"]
                for e in gutenberg
                if e["mainsnak"]["snaktype"] != "novalue"
            ]
            isbn = data["claims"].get("P212", [])
            isbns = [
                e["mainsnak"]["datavalue"]["value"]
                for e in isbn
                if e["mainsnak"]["snaktype"] != "novalue"
            ]
            if len(gutenberg_ids) > 0:
                has_gutenberg += 1
            if len(isbns) > 0:
                has_isbn += 1
                print(has_isbn)
            if len(genres) > 0:
                genre_counter.update(genre_ids)
            else:
                genre_counter.update([None])
            is_instance_claims = data["claims"]["P31"]
            is_instance_target_ids = [
                e["mainsnak"]["datavalue"]["value"]["id"] for e in is_instance_claims
            ]
            is_movie = "Q11424" in is_instance_target_ids
            is_book = "Q7725634" in is_instance_target_ids
            if is_book:
                book_count += 1
            if is_movie:
                movie_count += 1
            if is_movie and is_book:
                both_count += 1
            if not is_movie and not is_book:
                neither_count += 1
                print(story.description)
                print(story.wikidata_id)
            count += 1
        return {
            "neither_count": neither_count,
            "story_count": count,
            "num_books": book_count,
            "num_movies": movie_count,
            "num_both": both_count,
            "genres": genre_counter.most_common(),
            "has_gutenberg": has_gutenberg,
            "has_isbn": has_isbn,
        }

    def get_lang_stats(self):
        counter = Counter()
        counter_no_duplicates = Counter()
        length_counter = defaultdict(Counter)
        i = 0
        for story in tqdm(self.stories.values()):
            counter_no_duplicates.update(story.remove_duplicates().keys())
            counter.update(story.summaries_original.keys())
            i += 1
        return {
            "languages": dict(counter),
            "languages_direct_translations_removed": dict(counter_no_duplicates),
            "lengths_per_language": dict(length_counter),
        }


def pair_combinations(iterable):
    out = []
    for i, a in enumerate(iterable):
        for j, b in enumerate(iterable):
            if i >= j:
                continue
            else:
                out.append((a, b))
    return out


class SimilarityDataset:
    def __init__(
        self,
        data_path: Optional[Path] = None,
        anonymized=True,
        min_sentences=0,
        negative_sample_scale=1.0,
        seed=42,
        min_length=0,
    ):
        if data_path is None:
            data_path = download_dataset()
        self.summary_dataset = StoryDataset(data_path)
        splits = self.summary_dataset.perform_splits()
        self.summaries = {}
        randomizer = random.Random(seed)
        self.splits = {}
        for split in ["train", "dev", "test"]:
            if anonymized:
                summaries_getter = lambda x, min_length: x.get_anonymized(
                    min_sentences=min_length
                ).values()
            else:
                summaries_getter = lambda x, min_length: v.get_all_summaries_en(
                    min_sentences=min_length
                )[1]
            positive_samples = list(
                itertools.chain.from_iterable(
                    [
                        pair_combinations(summaries_getter(story, min_length))
                        for story in splits[split].stories.values()
                    ]
                )
            )
            num_negative_samples = int(len(positive_samples) * negative_sample_scale)
            negative_samples = []
            stories = list(splits[split].stories.values())
            for _ in range(num_negative_samples):
                story_a = randomizer.choice(stories)
                story_b = None
                while story_b == story_a or story_b is None:
                    story_b = randomizer.choice(stories)
                negative_samples.append(
                    (
                        randomizer.choice(list(story_b.get_anonymized().values())),
                        randomizer.choice(list(story_b.get_anonymized().values())),
                    )
                )
            negative_samples = [
                {"text_a": sample[0], "text_b": sample[1], "label": -1}
                for sample in negative_samples
            ]
            positive_samples = [
                {"text_a": sample[0], "text_b": sample[1], "label": 1}
                for sample in positive_samples
            ]
            samples = negative_samples + positive_samples
            randomizer.shuffle(samples)
            self.splits[split] = Dataset.from_list(samples)

    def __getitem__(self, split):
        return self.splits[split]


class Split:
    def __init__(self, items):
        self.samples = items

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        yield from self.samples

    def __getitem__(self, i):
        return self.samples[i]
