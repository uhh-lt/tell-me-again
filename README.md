# Tell me again!

A dataset of multiple summaries for the same story.

The summaries are scraped from Wikipedia and come in various languages.
We also provide automatically created translation and anonymzied versions of the summaries (without identifiable names).

## Usage
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

Optionally you can download the file from our website.
