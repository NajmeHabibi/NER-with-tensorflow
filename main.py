from database import NewsDatabase
from ners import SpacyNER, BertNER, LstmNER
from crawler import NewsCrawler


if __name__ == '__main__':
    # ner = SpacyNER()
    # ner = BertNER()
    ner = LstmNER()

    db = NewsDatabase()

    crawler = NewsCrawler()
    for doc in crawler.iter_news(limit=5):
        doc.entities = list(ner(text=doc.text))
        db.insert_into_tables(doc=doc)

