import psycopg2


class NamedEntity:
    def __init__(self, entity, start_char, end_char, label):
        self.entity = entity
        self.start_char = start_char
        self.end_char = end_char
        self.label = label

    def __str__(self):
        return f'NamedEntity(entity="{self.entity}", label={self.label}, start={self.start_char}, end={self.end_char})'


class Document:
    def __init__(self, text, title, entities=None):
        self.text = text
        self.title = title
        self.entities = entities


class NewsDatabase:
    NEXT_DOC_ID = 1

    def __init__(self):
        self.doc_table_name = 'document'
        self.ne_table_name = 'named_entity'

        self.conn = psycopg2.connect(
            database="postgres", user='postgres', password='postgres', host='localhost', port='5432'
        )
        self.conn.autocommit = True

        self.create_tables()

    def create_tables(self):
        sql = f"""
        DROP TABLE IF EXISTS {self.doc_table_name} CASCADE;
        DROP TABLE IF EXISTS {self.ne_table_name} CASCADE;

        CREATE TABLE {self.doc_table_name} (
            id             SERIAL PRIMARY KEY,
            title          VARCHAR NOT NULL,
            text           TEXT NOT NULL
        );

        CREATE TABLE {self.ne_table_name} (
            id             SERIAL PRIMARY KEY,
            entity         VARCHAR NOT NULL,
            start_char     INT NOT NULL,
            end_char       INT NOT NULL,
            label          VARCHAR NOT NULL,
            doc_id         INT REFERENCES {self.doc_table_name}(id) NOT NULL
        );
        """
        print(sql)

        print("Creating tables ...")
        cursor = self.conn.cursor()
        cursor.execute(sql)
        cursor.close()

    def insert_into_tables(self, doc):
        def _escape_single_quote(s):
            return s.replace("'", "''")

        doc_id = self.NEXT_DOC_ID
        doc_title = _escape_single_quote(doc.title)
        doc_text = _escape_single_quote(doc.text)
        doc_value_str = f"({doc_id}, '{doc_title}', '{doc_text}')"

        ne_values = []
        for ne in doc.entities:
            entity = _escape_single_quote(ne.entity)
            ne_values.append(f"('{entity}', {ne.start_char}, {ne.end_char}, '{ne.label}', {doc_id})")
        ne_values_str = ', '.join(ne_values)

        doc_sql = f"""INSERT INTO {self.doc_table_name} VALUES {doc_value_str};"""

        print("Inserting into tables ...")
        cursor = self.conn.cursor()
        cursor.execute(doc_sql)
        if ne_values_str != '':
            ent_sql = f"""INSERT INTO {self.ne_table_name} (entity, start_char, end_char, label, doc_id) VALUES {ne_values_str};"""
            cursor.execute(ent_sql)
        cursor.close()

        self.NEXT_DOC_ID += 1