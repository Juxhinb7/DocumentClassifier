import string


class TextVectorizer:
    def __init__(self):
        self.text: str = ""
        self.stop_words: [] = []
        self.list_of_words: [] = []

    def transform(self, text: str):
        self.text = text.lower().rstrip().translate(str.maketrans("", "", string.punctuation)).replace("•", "")
        with open("stopwords.txt", "r", encoding="utf-8") as sw:
            lines = sw.readlines()
            for line in lines:
                self.stop_words.append(line.rstrip())
        self.list_of_words = " ".join([w for w in self.text.split() if w not in self.stop_words]).split()
        return self.list_of_words

