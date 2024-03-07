# Tell me again!

A dataset of multiple summaries for the same story.

The summaries are scraped from Wikipedia and come in various languages.
We also provide automatically created translation and anonymized versions of the summaries (without identifiable names).


When using this dataset please cite the following paper:
```
@inproceedings{hatzel-etal-2024-tellme,
    title = "Tell me again! A Large-Scale Dataset of Multiple Summaries for the Same Story"
    booktitle = "Proceedings of the Forteenth Language Resources and Evaluation Conference",
    author = "Hatzel, Hans Ole and Biemann, Chris",
    address = "Turin, Italy",
    year = "2024"
}
```


## Usage
Install by running: `pip install tell_me_again`

```python
from tell_me_again import StoryDataset, SimilarityDataset

dataset = StoryDataset()
for story in dataset:
    # This prints a dictionary of the various anonymized versions of the story
    print(story.summaries_anonymized)

dataset = SimilarityDataset()
for pair in dataset["dev"]:
    print(pair["text_a"], pair["text_b"], pair["label"])
```

Optionally you can manually download the file from [our website](https://ltdata1.informatik.uni-hamburg.de/tell_me_again_v1.zip).
Point the `data_path` argument in the dataset classes to the zip file (with out extracting it) to use the manually downloaded copy.
