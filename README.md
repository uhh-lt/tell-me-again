# Tell me again!

A dataset of multiple summaries for the same story.

The summaries are scraped from Wikipedia and come in various languages.
We also provide automatically created translation and anonymized versions of the summaries (without identifiable names).
For further details, check out the corresponding paper [here](https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2024-hatzel-et-al-lrec.pdf).


When using this dataset please cite the following paper:
```
@inproceedings{hatzel-biemann-2024-tell-large,
    title = "Tell Me Again! a Large-Scale Dataset of Multiple Summaries for the Same Story",
    author = "Hatzel, Hans Ole and Biemann, Chris",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1366",
    pages = "15732--15741",
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

## License

The code in this repo is MIT licensed, the summaries were all scraped from Wikipedia and are Licensed (including their derived forms) under CC-BY-SA 4.0.
