# Python application saving extracted named entities of crawled news text corpora into a database:

* First, we implement a `crawler.NewsCrawler` class, crawling news from [CBC News website](https://www.cbc.ca/news) of different categories 
* Then, we use one of the three designed NER models, namely `ners.BertNER`, `ners.LstmNER`, and `ners.SpacyNER` (sorted here from higher to lower accuracy performance), to extract named entities of each crawled news text one by one in `main.py`
  * In `ners.BertNER`, we use the pretrained model `dslim/bert-base-NER` of `transformers` library, which is a fine-tuned version of BERT for NER task
  * In `ners.LstmNER`, we use our custom trained model `LSTM` under `lstm_ner.lstm_ner`, that is further explained there
  * In `ners.SpacyNER`, we use Spacy pipeline for NER
* After deriving named entities out of a crawled text (using previous two steps), finally, we use `database.NewsDatabase` class, implementing logics to create and insert into two tables of interest, 
  `document` table (with attributes `id, title, text`) and `named-entity` table (with attributes `id, entity, start_char, end_char, label, doc_id`), by connecting Python to PostgreSQL database using `psycopg2` connector library

### Run:
In a console with an activated python environment, under current directory:
* Enter `pip install -r requirements.txt` to install dependencies
  * As prerequisites, make sure to have PostgreSQL server installed, Selenium webdriver added to the path, and spacy data downloaded
* Enter `python main.py` to start crawling news text corpora, and put their extracted named entities into the database 