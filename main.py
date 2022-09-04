import PyPDF2



pdf = open("Meeting Minutes.pdf", "rb")
pdf_reader = PyPDF2.PdfFileReader(pdf)
pdf.close()
page_one = pdf_reader.getPage(0)

