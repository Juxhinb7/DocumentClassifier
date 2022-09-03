import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer


corpus = ['This is the first document.', 'This document is the second document.', 'And this is the third one.',
          'Is this the first document?']

pdf = open("Meeting Minutes.pdf", "rb")
pdf_reader = PyPDF2.PdfFileReader(pdf)
page_one = pdf_reader.getPage(0)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([page_one.extractText(), 'This document is the second document.'])
vectorizer.get_feature_names_out()
print(X)
