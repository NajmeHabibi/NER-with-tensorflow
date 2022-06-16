# Python application saving extracted named entities of crawled news text corpora into a database:

![ner](https://user-images.githubusercontent.com/42568062/174129808-20fc2fe8-1b32-4005-a036-999eaf2c6ced.png)

## Example
`Typical mortgage payment could be 30% higher in 5 years, Bank of Canada says the average price figure can be misleading because it is easily skewed by sales in large expensive markets such as Toronto and Vancouver. So it calculates another number, known as the House Price Index (HPI), which it says is a better gauge of the market because it adjusts for the volume and types of housing. Heaps Estrin, president and CEO of Toronto-based real estate, says the slowdown in Toronto is mostly happening in the suburbs, where prices jumped the most during the pandemic as buyers sought more space`

| Entity           | Label         | doc_id      |
| ---------------- | ------------- | --------    |
| `Bank of Canada` | `ORG`         | 3           |
| `Toronto`        | `LOC`         | 3           |
| `Vancouver`      | `LOC`         | 3           | 
| `HPI`            | `ORG`         | 3           |
| `Heaps Estrin`   | `PER`         | 3           |
| `Toronto`        | `LOC`         | 3           |

## Our Work
* First, we implement a `crawler.NewsCrawler` class, crawling news from [CBC News website](https://www.cbc.ca/news) of different categories 
* Then, we use one of the three designed NER models, namely `ners.BertNER`, `ners.LstmNER`, and `ners.SpacyNER` (sorted here from higher to lower accuracy performance), to extract named entities of each crawled news text one by one in `main.py`
  * In `ners.BertNER`, we use the pretrained model `dslim/bert-base-NER` of `transformers` library, which is a fine-tuned version of BERT for NER task
  * In `ners.LstmNER`, we use our custom trained model `LSTM` under `lstm_ner.lstm_ner`, that is further explained in [lstm_ner/Readme.md](https://github.com/NajmeHabibi/NER-with-tensorflow/tree/master/lstm_ner#readme)
  * In `ners.SpacyNER`, we use Spacy pipeline for NER
* After deriving named entities out of a crawled text (using previous two steps), finally, we use `database.NewsDatabase` class, implementing logics to create and insert into two tables of interest, 
  `document` table (with columns `id, title, text`) and `named-entity` table (with columns `id, entity, start_char, end_char, label, doc_id`, the `entity` column of which would be one of the three inferred types, namely `PER` for Person, `ORG` for Organization, and `LOC` for Location), 
   by connecting Python to PostgreSQL database using `psycopg2` connector library

### Run
In a console with an activated python environment, under current directory:
* Enter `pip install -r requirements.txt` to install dependencies
  * As prerequisites, make sure to have PostgreSQL server installed (equipped with the "postgres" user having password "postgres"), Selenium webdriver added to the path, and spacy data downloaded
* Enter `python main.py` to start crawling news text corpora, and put their extracted named entities into the database 
